"""
Phase 4 - Model Interpretation (Delivery 4, Part A.2)
=====================================================
Generates the model-interpretation figures required by Phase 4 that were NOT
produced in the Phase 3 notebook:

  1. Grad-CAM saliency maps for ResNet-18 (the selected model)  -> required for a CNN project
  2. (Optional) Attention-rollout maps for ViT-Small/4 and CCT-7/3x1

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
Phase 3 saved per-model checkpoints to Google Drive (e.g. ResNet-18_010pct_best.pt).
Interpretation must be run *against those checkpoints*, so it cannot be reproduced
from the notebook outputs alone. Run this in the same Colab/Drive environment that
produced Phase 3 so the model classes and checkpoint paths resolve.

HOW TO RUN
----------
1. Run this AFTER the Phase 3 model-definition cells (so `ResNet18CIFAR`, `ViT`,
   `CCT` and `CFG`, `CLASS_NAMES`, `test_loader`/test dataset are in scope), OR
   paste the model definitions above this file.
2. Set ROOT to your Phase 3 Drive folder.
3. Choose REGIME (0.10 is the most thesis-relevant low-data regime).
4. The script writes:
       gradcam_resnet18_<regime>.png
       attention_<model>_<regime>.png   (optional)
   Insert these into the thesis where marked [FIGURE PLACEHOLDER].

NOTE ON HONESTY
---------------
Do not describe the heatmaps before you have generated and inspected them.
The thesis text gives you a template; fill it from what you actually observe.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

# ----------------------------------------------------------------------------
# CONFIG - adapt to your environment
# ----------------------------------------------------------------------------
ROOT = "/content/drive/MyDrive/thesis/phase3"   # Phase 3 artefact folder
REGIME = 0.10                                    # 1.0, 0.10 or 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def regime_tag(frac):
    return f"{int(round(frac*100)):03d}pct"

# ----------------------------------------------------------------------------
# 1. GRAD-CAM for ResNet-18  (Selvaraju et al., 2017)
# ----------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM on a chosen convolutional layer.

    For a torchvision-style ResNet-18, `target_layer` is `model.layer4[-1]`
    (the last residual block). If your ResNet class names the trunk differently,
    point `target_layer` at the final conv stage instead.
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)                       # (1, num_classes)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        # weights = global-average-pooled gradients over the feature map
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, logits.softmax(dim=1)[0, class_idx].item()


def denormalize(t):
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def run_gradcam_resnet(model, dataset, n_examples=8, prefer_misclassified=True):
    """Pick a mix of correct and incorrect examples and overlay Grad-CAM."""
    cam_engine = GradCAM(model, target_layer=model.layer4[-1])
    chosen = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        x = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        wrong = (pred != label)
        if prefer_misclassified and wrong:
            chosen.append((idx, img, label, pred))
        elif not prefer_misclassified and not wrong:
            chosen.append((idx, img, label, pred))
        if len(chosen) >= n_examples:
            break
    # top up with whatever is available
    i = 0
    while len(chosen) < n_examples and i < len(dataset):
        img, label = dataset[i]
        chosen.append((i, img, label, None)); i += 1

    fig, axes = plt.subplots(2, n_examples, figsize=(2.0 * n_examples, 4.4))
    for col, (idx, img, label, pred) in enumerate(chosen):
        x = img.unsqueeze(0).to(DEVICE)
        cam, cls, conf = cam_engine(x, class_idx=None)
        rgb = denormalize(img)
        axes[0, col].imshow(rgb); axes[0, col].axis("off")
        axes[0, col].set_title(f"T:{CLASS_NAMES[label]}\nP:{CLASS_NAMES[cls]} ({conf:.2f})",
                               fontsize=7,
                               color="black" if cls == label else "crimson")
        axes[1, col].imshow(rgb); axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=8)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8)
    plt.suptitle(f"Grad-CAM - ResNet-18 @ {int(REGIME*100)}% "
                 f"(top: input, bottom: class-discriminative regions)", fontsize=10)
    plt.tight_layout()
    out = os.path.join(ROOT, f"gradcam_resnet18_{regime_tag(REGIME)}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
    print("saved", out)


# ----------------------------------------------------------------------------
# 2. (Optional) Attention rollout for the transformers (Abnar & Zuidema, 2020)
# ----------------------------------------------------------------------------
# Attention rollout requires capturing the per-layer attention matrices. The
# cleanest way is to add `return_attn` plumbing to your encoder block, OR register
# a forward hook on each block's softmax. Because the exact internal names depend
# on your Phase 3 class definitions, this is left as a documented stub: capture a
# list `attentions` of (heads, N, N) tensors, then:
#
#   def rollout(attentions):
#       result = torch.eye(attentions[0].size(-1))
#       for a in attentions:
#           a = a.mean(0)                      # average over heads
#           a = a + torch.eye(a.size(-1))      # add residual
#           a = a / a.sum(-1, keepdim=True)
#           result = a @ result
#       return result                          # (N, N); row 0 = CLS attention
#
# Map the CLS->patch row back onto the 8x8 (ViT) / token grid and upsample for
# an overlay, exactly as with Grad-CAM. For CCT, use the SeqPool attention
# weights directly (they already give one score per token).


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Expecting `ResNet18CIFAR`, `build_test_dataset()` (returning a non-shuffled
    # CIFAR-10 test Dataset with the *eval* transform) and `CFG` in scope.
    transform_eval = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=transform_eval)

    model = ResNet18CIFAR(num_classes=10).to(DEVICE)   # your Phase 3 class
    ckpt = os.path.join(ROOT, f"ResNet-18_{regime_tag(REGIME)}_best.pt")
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"] if "model" in state else state)
    print("loaded", ckpt)

    run_gradcam_resnet(model, test_ds, n_examples=8, prefer_misclassified=True)
