# thesis-cnn-vit-hybrid
# Comparing CNN, Vision Transformer, and Hybrid Architectures for Image Classification in Low-Data Regimes

Bachelor's thesis project — PJATK. Author: Ivan Tomilo.

This repository contains the code, trained-model artefacts, and analysis for a
controlled comparison of three architecture families on CIFAR-10 across three
training-data regimes (100%, 10%, 1%).

## Research question

Under a fixed, identical training budget, which architecture family is most
reliable when labelled training data is scarce — a convolutional network
(ResNet-18), a pure transformer (ViT-Small/4), or a convolution–transformer
hybrid (CCT-7/3×1)?

## Models

| Model | Family | Params | Key design point |
|-------|--------|--------|------------------|
| ResNet-18 | CNN | 11.17 M | CIFAR-adapted 3×3 stem (no 7×7 + maxpool) |
| ViT-Small/4 | Transformer | 21.34 M | 4×4 patches → 64 tokens + CLS, 12 blocks |
| CCT-7/3×1 | Hybrid | 3.77 M | conv tokenizer + sequence pooling, 7 blocks |

## Repository structure

```
.
├── README.md
├── notebooks/
│   └── PRO_3.ipynb              # Phase 3: training, comparison, error analysis
├── src/
│   └── phase4_interpretation.py # Phase 4: Grad-CAM + attention interpretation
├── artefacts/                   # produced by the notebook (saved to Drive)
│   ├── *_best.pt                # per-model, per-regime checkpoints
│   ├── *_results.json           # per-run metrics + predictions
│   ├── comparison_flat.csv
│   ├── comparison_pivot.csv
│   ├── confusion_matrices.png
│   ├── learning_curves_10pct.png
│   └── phase3_final_results.json
└── docs/
    └── Thesis.docx              # final thesis document
```

## How to reproduce

1. Open `notebooks/PRO_3.ipynb` in Google Colab with a GPU runtime (T4).
2. Mount Google Drive and set `ROOT` to your artefact folder.
3. Run all cells top to bottom. Section 11 trains all three models across all
   three regimes (~1.5–2 h on a T4 at 30 epochs) and writes checkpoints + JSON.
4. Sections 12.1–12.6 reproduce the comparison table, learning curves,
   confusion matrices, per-class metrics, McNemar tests, and model selection.
5. For Phase 4 interpretation, run `src/phase4_interpretation.py` in the same
   environment (it loads the saved checkpoints and writes the Grad-CAM figure).

## Fixed experimental configuration

AdamW (lr 5e-4, weight decay 0.05), cosine annealing with 3 linear-warmup epochs,
30 epochs, batch size 128, RandAugment (N=2, M=9), Mixup (α=0.8), CutMix (α=1.0),
mix probability 0.5, label smoothing 0.1, gradient clipping 1.0, AMP, seed 42,
stratified 90/10 train/val split. Only the architecture changes between runs.

## Headline result

ResNet-18 achieves the highest test accuracy at **every** regime
(92.64% / 74.54% / 40.43% at 100% / 10% / 1%), with CCT-7/3×1 second and
ViT-Small/4 last. No crossover point was observed within this training budget.

## Important caveat

The 30-epoch budget is below the convergence horizon for the transformer models
(the literature commonly uses 200–300 epochs from scratch). Results therefore
characterise behaviour under a *fixed, compute-constrained* budget rather than
fully converged performance. See the Discussion and Limitations section of the
thesis.

## License / academic integrity

Coursework submission for PJATK. Third-party methods are cited in the thesis
references.
