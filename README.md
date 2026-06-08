# ECG SHAP Explainability

SHAP-based post-hoc explainability for the **LightECGNetV2** 12-lead ECG
multi-label classifier.  This module lives inside the existing LightECGNet
repository and reuses the trained fold checkpoints produced by the main
training pipeline.

---

## Repository layout (after adding this module)

```
LightECGNet/                          ← existing repo root
│
├── models/                           ← existing: model architecture definitions
│   └── lightecgnetv2.py              ← LightECGNetV2, building blocks
│
├── train/                            ← existing: training scripts
│
├── checkpoints/                      ← existing: saved fold weights
│   ├── lightv2_ft_fold0.pt           ← fold-0 fine-tuned checkpoint
│   ├── lightv2_p1_fold1.pt           ← fold-1 phase-1 checkpoint
│   └── lightv2_p1_fold2.pt           ← fold-2 phase-1 checkpoint
│
├── dataset/                          ← existing or symlinked PhysioNet data
│   ├── WFDBRecords/                  ← .mat + .hea files
│   └── ConditionNames_SNOMED-CT.csv  ← SNOMED-CT code → acronym mapping
│
├── explainability/                   ← NEW: everything added by this PR
│   ├── ecg_shap_explainability.py    ← main script (this module)
│   ├── requirements.txt              ← extra deps (shap, scipy SOS)
│   └── README.md                     ← this file
│
└── outputs/                          ← auto-created at runtime; gitignored
    └── shap_*.png                    ← generated figures
```

---

## Files to add to the repo

| File | Where | Purpose |
|------|--------|---------|
| `explainability/ecg_shap_explainability.py` | new subfolder | Main SHAP script |
| `explainability/requirements.txt` | new subfolder | Extra pip deps |
| `explainability/README.md` | new subfolder | This document |
| `.gitignore` additions | repo root | Ignore `outputs/`, `class_names.npy`, `*.pt` (if not already) |

**Do NOT commit:**
- `checkpoints/*.pt` — large binary files; host on GDrive / HuggingFace Hub
- `dataset/` — PhysioNet data; download separately
- `outputs/shap_*.png` — generated artefacts
- `class_names.npy` — auto-generated cache; reproducible from `.hea` files

---

## Setup

```bash
# 1. Clone the repo (if not already done)
git clone https://github.com/<your-org>/LightECGNet.git
cd LightECGNet

# 2. Install base dependencies (from existing repo)
pip install -r requirements.txt

# 3. Install explainability extra dependencies
pip install -r explainability/requirements.txt

# 4. Download / symlink the PhysioNet dataset
#    Expected layout: dataset/WFDBRecords/**/*.mat + *.hea
#    and:             dataset/ConditionNames_SNOMED-CT.csv

# 5. Place checkpoint files (from training pipeline or shared drive)
#    Expected:  checkpoints/lightv2_ft_fold0.pt
#               checkpoints/lightv2_p1_fold1.pt
#               checkpoints/lightv2_p1_fold2.pt
```

---

## Running the explainability script

Run every cell block sequentially in Jupyter, or execute the whole script:

```bash
cd LightECGNet
python explainability/ecg_shap_explainability.py
```

Output PNG files are saved to the working directory (or `outputs/` if you
set `OUTPUT_DIR` in Cell 2).

---

## Checkpoint naming convention

| File | Training stage | Description |
|------|---------------|-------------|
| `lightv2_ft_fold0.pt` | Fine-tuned | Fold-0 of k-fold CV, full fine-tune |
| `lightv2_p1_fold1.pt` | Phase-1 | Fold-1 of k-fold CV, phase-1 weights |
| `lightv2_p1_fold2.pt` | Phase-1 | Fold-2 of k-fold CV, phase-1 weights |

All three are `state_dict` files (plain `torch.save(model.state_dict(), path)`).
They are loaded with `weights_only=True` for safety.

---

## What the script produces

| Output file | Cell | Description |
|-------------|------|-------------|
| `raw_ecg_12lead.png` | 9 | Preprocessed 12-lead ECG waveform |
| `shap_scatter_<CLASS>.png` | 14 | Red/blue SHAP dots on ECG per lead |
| `shap_heatmap_<CLASS>.png` | 15 | 2-D heatmap: 12 leads × 5000 time steps |
| `shap_lead_importance_<CLASS>.png` | 16 | Per-lead bar chart + LeadAttention overlay |
| `shap_pqrst_<CLASS>.png` | 17 | P-wave / QRS / T-wave attribution breakdown |
| `shap_temporal_overlay_<LEAD>.png` | 18 | Multi-class colour-coded temporal regions |
| `shap_population_heatmap_<CLASS>.png` | 20 | Population-averaged |SHAP| heatmap |
| `class_names.npy` | 6 | Cached class list (auto-generated, gitignored) |
