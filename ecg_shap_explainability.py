"""
explainability/ecg_shap_explainability.py
==========================================
SHAP-based explainability + ablation check for LightECGNetV2.

USAGE
-----
Run on a specific ECG file (recommended):
    python explainability/ecg_shap_explainability.py --input path/to/record.mat

Run on the first file found in DATASET_ROOT (fallback):
    python explainability/ecg_shap_explainability.py

All output PNGs are named after the input file so results are traceable:
    shap_highlighted_<record>_<CLASS>.png   -- SHAP-highlighted 12-lead ECG
    shap_ablated_<record>_<CLASS>.png       -- same ECG with ablated regions marked
    ablation_report_<record>.txt            -- ablation probability drop summary

HOW IT WORKS
------------
1. Load and preprocess the user-specified ECG (.mat file).
2. Load the 3-fold ensemble and score the ECG -> top-5 predicted classes.
3. Build a background tensor from OTHER files (input file is excluded).
4. Run GradientExplainer to get SHAP values for the top-5 classes.
5. For each top-5 class:
     a. Highlight the 12-lead ECG: red = positive SHAP, blue = negative SHAP,
        plotted on the actual input ECG waveform.
     b. Run ablation: zero out top-10% SHAP time steps, re-score, report drop.
     c. Save a second plot showing which exact time steps were zeroed out.

NOTE ON MODEL REUSE
--------------------
LightECGNetV2 is defined inline here for portability.
If models/lightecgnetv2.py is importable from the repo root, replace
the model definition section with:
    from models.lightecgnetv2 import (
        LightECGNetV2, EnsembleModel, load_single_model, load_ensemble
    )

Optimizations applied:
  [OPT-1]  SOS Butterworth filter pre-computed once; vectorised over all leads.
  [OPT-2]  ThreadPoolExecutor parallelises background .mat loading.
  [OPT-3]  torch.compile() on PyTorch >= 2.0.
  [OPT-4]  torch.inference_mode() replaces no_grad everywhere.
  [OPT-5]  torch.amp.autocast (float16) on CUDA.
  [OPT-6]  Background tensor excludes the input sample (no self-contamination).
  [OPT-7]  weights_only=True in torch.load.
  [OPT-8]  Matplotlib Agg backend.
"""

## cell 1 start
# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import os
import sys
import csv
import glob
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # [OPT-8] no display server needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt   # [OPT-1] SOS form

import torch
import torch.nn as nn
import torch.nn.functional as F

import shap

warnings.filterwarnings("ignore")
## cell 1 end


## cell 2 start
# -------------------------------------------------------------------
# Device setup and AMP (Automatic Mixed Precision) configuration
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

USE_AMP  = device.type == "cuda"
_amp_ctx = (torch.amp.autocast(device_type="cuda")
            if USE_AMP
            else torch.amp.autocast(device_type="cpu", enabled=False))
## cell 2 end


## cell 3 start
# -------------------------------------------------------------------
# Configuration
# Dataset and CSV paths are set to absolute paths for your machine.
# Checkpoints are resolved relative to this script's own directory,
# falling back to os.getcwd() when running inside a Jupyter notebook
# (where __file__ is not defined).
# -------------------------------------------------------------------
SIGNAL_LEN  = 5000
FS          = 500
NUM_CLASSES = 45

# Checkpoint paths — one .pt file per cross-validation fold.
MODEL_PATHS = [
    r"C:\Users\lenovo\HonsExAI\lightv2_ft_fold0.pt",
    r"C:\Users\lenovo\HonsExAI\lightv2_p1_fold1.pt",
    r"C:\Users\lenovo\HonsExAI\lightv2_p1_fold2.pt",
]

# Absolute paths to the dataset and condition-name mapping.
DATASET_ROOT = r"C:\Users\lenovo\HonsExAI\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords"
MAPPING_CSV  = r"C:\Users\lenovo\HonsExAI\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv"

# Quick sanity-check — print resolved paths so you can spot mistakes early.
print(f"DATASET_ROOT : {DATASET_ROOT}  (exists: {os.path.isdir(DATASET_ROOT)})")
print(f"MAPPING_CSV  : {MAPPING_CSV}  (exists: {os.path.isfile(MAPPING_CSV)})")
for p in MODEL_PATHS:
    print(f"  checkpoint : {p}  (exists: {os.path.isfile(p)})")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

COLORS_TOP5 = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
## cell 3 end


## cell 4 start
# -------------------------------------------------------------------
# Preprocessing  [OPT-1]
# Pre-compute the bandpass SOS filter once; apply vectorised over all leads.
# Must be identical to the preprocessing used during training.
# -------------------------------------------------------------------
_BP_SOS = butter(3, [0.5 / (0.5 * FS), 40.0 / (0.5 * FS)],
                 btype="band", output="sos")


def preprocess_ecg_mat(mat_path, target_len=SIGNAL_LEN):
    """
    Load a PhysioNet .mat ECG and apply training-identical preprocessing:
      1. Read 'val' or 'data' key
      2. Ensure (12, T) shape
      3. Pad or crop to target_len
      4. Bandpass filter all 12 leads (0.5-40 Hz) vectorised  [OPT-1]
      5. Per-lead z-score normalise

    Returns np.ndarray (12, target_len) float32.
    """
    mat    = loadmat(mat_path)
    signal = mat.get("val", mat.get("data"))
    if signal is None:
        raise ValueError(f"No ECG signal found in: {mat_path}")
    if signal.shape[0] != 12:
        signal = signal.T

    cur_len = signal.shape[1]
    if cur_len < target_len:
        signal = np.concatenate(
            [signal, np.zeros((12, target_len - cur_len), dtype=signal.dtype)],
            axis=1
        )
    elif cur_len > target_len:
        signal = signal[:, :target_len]

    signal = sosfiltfilt(_BP_SOS, signal.astype(np.float32), axis=-1)
    mean   = signal.mean(axis=1, keepdims=True)
    std    = signal.std(axis=1,  keepdims=True) + 1e-8
    return ((signal - mean) / std).astype(np.float32)
## cell 4 end


## cell 5 start
# -------------------------------------------------------------------
# Model building blocks
# Replicated from models/lightecgnetv2.py — keep in sync with that file.
# -------------------------------------------------------------------

class LeadAttention(nn.Module):
    def __init__(self, n_leads=12, reduction=2):
        super().__init__()
        hidden   = max(1, n_leads // reduction)
        self.fc1 = nn.Linear(n_leads, hidden)
        self.fc2 = nn.Linear(hidden,  n_leads)

    def forward(self, x):
        w = x.mean(dim=-1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)


class DSConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad      = kernel // 2
        self.dw  = nn.Conv1d(in_ch, in_ch,  kernel, padding=pad,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm1d(out_ch)
        self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden   = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden,   channels)

    def forward(self, x):
        w = x.mean(dim=-1)
        w = F.relu(self.fc1(w))
        return x * torch.sigmoid(self.fc2(w)).unsqueeze(-1)


class DilatedTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=13, dilation=1, drop=0.1):
        super().__init__()
        pad       = ((kernel - 1) * dilation) // 2
        self.dw   = nn.Conv1d(in_ch, in_ch, kernel,
                              padding=pad, dilation=dilation,
                              groups=in_ch, bias=False)
        self.pw   = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn1  = nn.BatchNorm1d(in_ch)
        self.bn2  = nn.BatchNorm1d(out_ch)
        self.se   = SEBlock(out_ch)
        self.drop = nn.Dropout(drop)
        self.skip = (nn.Identity() if in_ch == out_ch else
                     nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                   nn.BatchNorm1d(out_ch)))
        self.act  = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.drop(self.se(self.bn2(self.pw(self.bn1(self.dw(x))))))
        return self.act(out + identity)
## cell 5 end


## cell 6 start
# -------------------------------------------------------------------
# LightECGNetV2 — main model definition
# Lightweight multi-scale TCN for 12-lead ECG multi-label classification.
# -------------------------------------------------------------------

class LightECGNetV2(nn.Module):
    """
    Lightweight multi-scale TCN for 12-lead ECG multi-label classification.
    Input : (B, 12, 5000) float32 preprocessed ECG
    Output: (B, 45) raw logits — sigmoid applied externally
    """
    def __init__(self, num_classes=45, n_leads=12):
        super().__init__()
        self.lead_attn  = LeadAttention(n_leads, reduction=2)
        self.stem_k3    = DSConv1d(n_leads, 16, kernel=3)
        self.stem_k7    = DSConv1d(n_leads, 16, kernel=7)
        self.stem_k15   = DSConv1d(n_leads, 32, kernel=15)
        self.stem_proj  = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.PReLU()
        )
        self.stem_pool  = nn.AvgPool1d(2)
        self.tcn1a      = DilatedTCNBlock(64,  64,  kernel=13, dilation=1, drop=0.1)
        self.tcn1b      = DilatedTCNBlock(64,  64,  kernel=13, dilation=2, drop=0.1)
        self.pool1      = nn.AvgPool1d(2)
        self.tcn2a      = DilatedTCNBlock(64,  128, kernel=13, dilation=4, drop=0.2)
        self.tcn2b      = DilatedTCNBlock(128, 128, kernel=13, dilation=8, drop=0.2)
        self.pool2      = nn.AvgPool1d(2)
        self.tcn3a      = DilatedTCNBlock(128, 256, kernel=9,  dilation=1, drop=0.3)
        self.tcn3b      = DilatedTCNBlock(256, 256, kernel=9,  dilation=2, drop=0.3)
        self.head       = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),   # inplace=False required for SHAP hooks
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.lead_attn(x)
        x = torch.cat([self.stem_k3(x), self.stem_k7(x), self.stem_k15(x)], dim=1)
        x = self.stem_pool(self.stem_proj(x))
        x = self.pool1(self.tcn1b(self.tcn1a(x)))
        x = self.pool2(self.tcn2b(self.tcn2a(x)))
        x = self.tcn3b(self.tcn3a(x))
        feat = torch.cat([x.max(dim=-1).values, x.mean(dim=-1)], dim=1)
        return self.head(feat)
## cell 6 end


## cell 7 start
# -------------------------------------------------------------------
# EnsembleModel + load_ensemble + AMP wrapper
# Averages sigmoid probabilities from 3 fold checkpoints.
# -------------------------------------------------------------------

class EnsembleModel(nn.Module):
    """
    Averages sigmoid probabilities from 3 fold checkpoints.
    Uses sigmoid (correct for multi-label) — not softmax.
    Input : (B, 12, 5000)
    Output: (B, 45) averaged sigmoid probabilities
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        probs = [torch.sigmoid(m(x)) for m in self.models]
        return torch.stack(probs, dim=0).mean(dim=0)


def load_ensemble(model_paths):
    """
    Load all fold checkpoints into EnsembleModel.
    [OPT-7] weights_only=True for safe, fast loading.
    [OPT-3] torch.compile() on PyTorch >= 2.0.
    """
    models = []
    for path in model_paths:
        print(f"  Loading: {path}")
        m = LightECGNetV2(num_classes=NUM_CLASSES).to(device)
        state = torch.load(path, map_location=device, weights_only=False)  # [OPT-7] weights_only=False: checkpoints are OrderedDicts saved without SafeTensor format
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)

    ensemble = EnsembleModel(models).to(device)
    ensemble.eval()

    if int(torch.__version__.split(".")[0]) >= 2 and device.type == "cuda":  # [OPT-3] compile only on CUDA; Windows CPU dynamo is unreliable
        try:
            ensemble = torch.compile(ensemble)
            print("  [OPT-3] torch.compile() applied.")
        except Exception as e:
            print(f"  [OPT-3] torch.compile() skipped: {e}")

    print(f"Ensemble ready — {len(models)} fold models.")
    return ensemble


class _AmpWrapper(nn.Module):
    """Wraps ensemble_model in autocast [OPT-5]; returns float32 for SHAP."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with _amp_ctx:
            return self.model(x).float()
## cell 7 end


## cell 8 start
# -------------------------------------------------------------------
# Class name loader
# Reads SNOMED-CT mapping CSV and scans .hea files to build class list.
# -------------------------------------------------------------------

def load_class_names(mapping_csv, dataset_root, exclude=None):
    cache = "class_names.npy"
    if os.path.exists(cache):
        names = np.load(cache, allow_pickle=True).tolist()
        print(f"Loaded class_names from cache ({len(names)} classes).")
        return names

    if exclude is None:
        exclude = {'ABI', 'VET', 'FQRS', 'SAAWR', 'JPT', 'VB'}

    snomed_to_acronym = {}
    with open(mapping_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            snomed_to_acronym[row["Snomed_CT"].strip()] = row["Acronym Name"].strip()

    def _dx(hea):
        try:
            with open(hea) as f:
                for line in f:
                    if "#Dx" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            return [c.strip() for c in parts[1].split(",") if c.strip()]
        except Exception:
            pass
        return []

    counts    = Counter()
    hea_files = glob.glob(os.path.join(dataset_root, "**", "*.hea"), recursive=True)
    for hea in hea_files:
        for code in _dx(hea):
            if code in snomed_to_acronym:
                name = snomed_to_acronym[code]
                if name not in exclude:
                    counts[name] += 1

    names = sorted(counts.keys())
    np.save(cache, names)
    print(f"Saved class_names.npy ({len(names)} classes).")
    return names
## cell 8 end


## cell 9 start
# -------------------------------------------------------------------
# Background tensor builder  [OPT-2, OPT-6]
# Loads n_bg real ECG samples in parallel, explicitly excluding the input file.
# -------------------------------------------------------------------

def build_background_tensor(mat_files, input_path, n_bg=8, max_workers=8):
    """
    Load n_bg real ECG samples, explicitly excluding input_path.

    [OPT-6] The input sample must NOT appear in its own SHAP background.
    Including it would contaminate the expected-value baseline, biasing all
    attributions toward zero for the regions most characteristic of that sample.
    [OPT-2] Parallel loading via ThreadPoolExecutor.

    Returns torch.Tensor (n_bg, 12, SIGNAL_LEN) on device.
    """
    input_abs  = os.path.abspath(input_path)
    # [OPT-6] remove input file from background candidates
    candidates = [p for p in mat_files if os.path.abspath(p) != input_abs]

    bg_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:  # [OPT-2]
        futures = {pool.submit(preprocess_ecg_mat, p): p
                   for p in candidates[:n_bg * 4]}
        for fut in as_completed(futures):
            if len(bg_list) >= n_bg:
                break
            try:
                bg_list.append(fut.result())
            except Exception as e:
                print(f"  Skipping {futures[fut]}: {e}")

    if not bg_list:
        raise RuntimeError("Could not load any background samples.")

    bg_np = np.stack(bg_list[:n_bg])
    bg_t  = torch.tensor(bg_np, dtype=torch.float32).to(device)
    print(f"Background: {bg_t.shape}  (input excluded  [OPT-6])")
    return bg_t
## cell 9 end


## cell 10 start
# -------------------------------------------------------------------
# Plot 1: SHAP-highlighted input ECG
# The actual user-supplied ECG with red/blue attribution dots.
# Red  = positive SHAP (supports prediction), Blue = negative SHAP (opposes it).
# -------------------------------------------------------------------

def plot_shap_highlighted_ecg(sample_np, shap_3d, explained_idx,
                               class_names, probs, time_axis,
                               record_name, explained_rank=0,
                               percentile=90, output_dir="."):
    """
    Plot all 12 leads of the INPUT ECG with SHAP attributions highlighted.

    The ECG waveform shown is the exact file passed via --input.
    SHAP dots are computed from that same file's gradients — not a proxy.

      Red  dots = positive SHAP (time steps that push the model TOWARD
                  predicting this class for this patient's ECG)
      Blue dots = negative SHAP (time steps that push the model AWAY
                  from predicting this class for this patient's ECG)

    Only the top (100-percentile)% magnitude SHAP values are shown
    to keep the plot readable.

    Output: shap_highlighted_<record>_<CLASS>.png
    """
    sv       = shap_3d[explained_rank]   # (12, 5000) for this class
    cls_idx  = explained_idx[explained_rank]
    cls_name = class_names[cls_idx]
    prob_val = probs[cls_idx]

    fig, axes = plt.subplots(12, 1, figsize=(18, 22), sharex=True)

    for i, ax in enumerate(axes):
        y        = sample_np[i]          # actual input ECG lead i
        s        = sv[i]                 # SHAP values for this lead

        thr      = np.percentile(np.abs(s), percentile)
        pos_mask = (np.abs(s) >= thr) & (s > 0)
        neg_mask = (np.abs(s) >= thr) & (s < 0)

        # The input ECG waveform in black
        ax.plot(time_axis, y, color="black", lw=0.9, alpha=0.95, zorder=1)

        # SHAP attribution dots overlaid on the same waveform
        ax.scatter(time_axis[pos_mask], y[pos_mask],
                   color="red",  s=16, alpha=1.0, edgecolors="none", zorder=3)
        ax.scatter(time_axis[neg_mask], y[neg_mask],
                   color="blue", s=16, alpha=1.0, edgecolors="none", zorder=3)

        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=22, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_xlim(0, SIGNAL_LEN / FS)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    legend_items = [
        Line2D([0], [0], color="black", lw=1.2, label="ECG waveform"),
        mpatches.Patch(facecolor="red",
                       label=f"Positive SHAP — supports '{cls_name}' "
                             f"(top {100-percentile:.0f}% |SHAP|)"),
        mpatches.Patch(facecolor="blue",
                       label=f"Negative SHAP — opposes '{cls_name}' "
                             f"(top {100-percentile:.0f}% |SHAP|)"),
    ]
    fig.legend(handles=legend_items, loc="upper right",
               bbox_to_anchor=(0.98, 0.99), frameon=True, fontsize=9)
    fig.suptitle(
        f"SHAP-Highlighted ECG  —  {record_name}\n"
        f"Predicted: {cls_name}   p = {prob_val:.4f}   "
        f"(rank #{explained_rank + 1} of top-{len(explained_idx)})",
        fontsize=13, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 0.97, 0.990])

    out_path = os.path.join(output_dir,
                            f"shap_highlighted_{record_name}_{cls_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path
## cell 10 end


## cell 11 start
# -------------------------------------------------------------------
# Plot 2: Ablated input ECG
# The same input ECG with zeroed (ablated) regions shaded in grey.
# Grey bands = top SHAP time steps set to 0 to test causal relevance.
# -------------------------------------------------------------------

def plot_ablated_ecg(sample_np, ablate_mask_2d, cls_name,
                     prob_before, prob_after,
                     time_axis, record_name, output_dir="."):
    """
    Plot all 12 leads of the INPUT ECG with the ablated time regions
    shaded in grey.

    ablate_mask_2d : (12, 5000) bool — per-lead mask; each lead has its OWN
                     set of grey bands computed from that lead's |SHAP| values.
    Grey bands = time steps that were zeroed out (set to 0.0 = z-score mean).
    These are the regions SHAP flagged as most important for that specific lead.
    The title shows the probability drop to confirm causal relevance.

    Output: shap_ablated_<record>_<CLASS>.png
    """
    delta     = prob_before - prob_after
    pass_fail = "PASS \u2713" if delta > 0.05 else "WARN \u2717"

    fig, axes = plt.subplots(12, 1, figsize=(18, 22), sharex=True)

    for i, ax in enumerate(axes):
        y    = sample_np[i]          # actual input ECG lead i
        mask = ablate_mask_2d[i]     # per-lead boolean mask (5000,)

        # Input ECG waveform in black
        ax.plot(time_axis, y, color="black", lw=0.9, zorder=2)

        # Grey shading over THIS LEAD's ablated (zeroed) time regions
        in_region, start_t = False, 0
        for t in range(len(mask)):
            if mask[t] and not in_region:
                start_t   = t
                in_region = True
            elif not mask[t] and in_region:
                ax.axvspan(time_axis[start_t], time_axis[t],
                           alpha=0.45, color="grey", lw=0, zorder=1)
                in_region = False
        if in_region:
            ax.axvspan(time_axis[start_t], time_axis[-1],
                       alpha=0.45, color="grey", lw=0, zorder=1)

        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=22, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_xlim(0, SIGNAL_LEN / FS)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    legend_items = [
        Line2D([0], [0], color="black", lw=1.2, label="ECG waveform"),
        mpatches.Patch(facecolor="grey", alpha=0.45,
                       label="Ablated region (per-lead top 10% |SHAP| — zeroed to 0)"),
    ]
    fig.legend(handles=legend_items, loc="upper right",
               bbox_to_anchor=(0.98, 0.99), frameon=True, fontsize=9)
    fig.suptitle(
        f"Ablation Check  —  {record_name}  —  Class: {cls_name}\n"
        f"p before ablation: {prob_before:.4f}   ->   "
        f"p after ablation: {prob_after:.4f}   "
        f"drop: {delta:+.4f} ({delta / prob_before * 100:.1f}%)   "
        f"{pass_fail}",
        fontsize=12, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 0.97, 0.990])

    out_path = os.path.join(output_dir,
                            f"shap_ablated_{record_name}_{cls_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path
## cell 11 end


## cell 12 start
# -------------------------------------------------------------------
# Ablation logic
# Zeroes out top-|SHAP| time steps, re-scores, and reports confidence drop.
# -------------------------------------------------------------------

def run_ablation(sample_np, ensemble_model, shap_3d, explained_idx,
                 class_names, probs, time_axis, record_name,
                 explained_rank=0, ablate_percentile=90, output_dir="."):
    """
    Per-lead ablation: for each of the 12 leads, zero out that lead's own
    top-(100-ablate_percentile)% |SHAP| time steps, then re-score the ECG.

    Using per-lead thresholds means each lead's grey bands are determined by
    that lead's own SHAP distribution — different leads highlight different
    temporal windows, as expected clinically.

    Returns
    -------
    delta          : float          -- probability drop (positive = fell)
    ablate_mask_2d : (12, 5000) bool -- per-lead mask of zeroed time steps
    """
    cls_idx   = explained_idx[explained_rank]
    cls_name  = class_names[cls_idx]
    orig_prob = probs[cls_idx]

    sv = shap_3d[explained_rank]          # (12, 5000)

    # --- Per-lead mask: each lead uses its own |SHAP| percentile threshold ---
    ablate_mask_2d = np.zeros((12, SIGNAL_LEN), dtype=bool)
    for lead_i in range(12):
        lead_attr              = np.abs(sv[lead_i])             # (5000,)
        thr                    = np.percentile(lead_attr, ablate_percentile)
        ablate_mask_2d[lead_i] = lead_attr >= thr               # top 10% for this lead

    # Zero out each lead's own flagged time steps (0 = z-score mean)
    ablated_np = sample_np.copy()
    for lead_i in range(12):
        ablated_np[lead_i, ablate_mask_2d[lead_i]] = 0.0

    ablated_tensor = torch.tensor(ablated_np).unsqueeze(0).to(device)
    with torch.inference_mode(), _amp_ctx:         # [OPT-4, OPT-5]
        ablated_prob = ensemble_model(ablated_tensor)[0, cls_idx].float().item()

    delta = orig_prob - ablated_prob
    total_zeroed = int(ablate_mask_2d.sum())   # total (lead, time) cells zeroed

    print(f"\n  {'='*52}")
    print(f"  Ablation  --  {cls_name}")
    print(f"    Original probability : {orig_prob:.4f}")
    print(f"    Ablated  probability : {ablated_prob:.4f}")
    print(f"    Probability drop     : {delta:+.4f}  "
          f"({delta / orig_prob * 100:.1f}%)")
    print(f"    Cells zeroed         : {total_zeroed} / {12 * SIGNAL_LEN} "
          f"(per-lead top {100-ablate_percentile:.0f}% |SHAP|)")
    if delta > 0.05:
        print(f"    PASS -- highlighted regions are causally relevant.")
    else:
        print(f"    WARN -- small drop; may be non-causal features.")
    print(f"  {'='*52}")

    # Plot the input ECG with per-lead grey shading
    plot_ablated_ecg(
        sample_np, ablate_mask_2d, cls_name,
        orig_prob, ablated_prob,
        time_axis, record_name, output_dir=output_dir
    )

    return delta, ablate_mask_2d
## cell 12 end


## cell 13 start
# -------------------------------------------------------------------
# Argument parser
# Defines CLI flags: --input, --top_k, --percentile, --ablate_percentile,
# --output_dir, --n_bg.
# NOTE: In a notebook, skip parse_args() and instead set these variables
# directly in cell 14 before calling main() or the pipeline steps.
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP explainability + ablation for LightECGNetV2"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help=(
            "Path to the .mat ECG file to explain. "
            "Example: --input dataset/WFDBRecords/A/A001/A00001.mat"
        )
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predicted classes to explain (default: 5)."
    )
    parser.add_argument(
        "--percentile", type=float, default=90.0,
        help="SHAP threshold percentile for dot highlighting (default: 90)."
    )
    parser.add_argument(
        "--ablate_percentile", type=float, default=90.0,
        help="Percentile above which time steps are zeroed in ablation (default: 90)."
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=".",
        help="Directory to save output PNGs (default: current directory)."
    )
    parser.add_argument(
        "--n_bg", type=int, default=8,
        help="Number of background samples for SHAP (default: 8)."
    )
    return parser.parse_args()
## cell 13 end


## cell 14 start
# -------------------------------------------------------------------
# Main pipeline - Steps 1-5
# Resolve input ECG path, preprocess the signal, load class names,
# load the ensemble model, and score the ECG to get top-K predictions.
# -------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Step 1: Resolve the input ECG path
    mat_files = glob.glob(
        os.path.join(DATASET_ROOT, "**", "*.mat"), recursive=True
    )
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found under {DATASET_ROOT}. "
            "Check DATASET_ROOT at the top of this script."
        )

    if args.input is not None:
        # User explicitly specified the ECG to explain
        input_path = args.input
        if not os.path.isfile(input_path):
            raise FileNotFoundError(
                f"Input file not found: {input_path}\n"
                "Pass a valid path with --input path/to/record.mat"
            )
    else:
        # No --input given: fall back to first file, but warn loudly
        input_path = mat_files[0]
        print(
            "\n  WARNING: No --input specified.\n"
            f"  Falling back to: {input_path}\n"
            "  For meaningful results always specify your target ECG:\n"
            "    python explainability/ecg_shap_explainability.py "
            "--input <your_ecg.mat>\n"
        )

    record_name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"\n{'='*60}")
    print(f"  Input ECG : {input_path}")
    print(f"  Record    : {record_name}")
    print(f"{'='*60}\n")

    # -- Step 2: Preprocess the input ECG
    print("Preprocessing input ECG ...")
    sample_np     = preprocess_ecg_mat(input_path)                   # (12, 5000)
    sample_tensor = torch.tensor(sample_np).unsqueeze(0).to(device)  # (1, 12, 5000)
    time_axis     = np.arange(SIGNAL_LEN) / FS                       # seconds

    # -- Step 3: Load class names
    class_names = load_class_names(MAPPING_CSV, DATASET_ROOT)

    # -- Step 4: Load ensemble
    print("\nLoading ensemble ...")
    ensemble_model = load_ensemble(MODEL_PATHS)

    # Warm-up pass: triggers torch.compile() JIT before timed inference [OPT-3]
    _dummy = torch.randn(1, 12, SIGNAL_LEN).to(device)
    with torch.inference_mode(), _amp_ctx:
        ensemble_model(_dummy)
    del _dummy

    # -- Step 5: Score the input ECG
    print(f"\nScoring {record_name} ...")
    with torch.inference_mode(), _amp_ctx:                            # [OPT-4, OPT-5]
        probs = ensemble_model(sample_tensor)[0].float().cpu().numpy()  # (45,)

    top_k_idx = np.argsort(probs)[::-1][:args.top_k]
    print(f"\nTop {args.top_k} predictions for  {record_name}:")
    for rank, idx in enumerate(top_k_idx, 1):
        print(f"  {rank}. {class_names[idx]:10s}  p = {probs[idx]:.4f}")
## cell 14 end


## cell 15 start
# -------------------------------------------------------------------
# Main pipeline - Steps 6-8
# Build SHAP background tensor (excluding input), initialise
# GradientExplainer, and compute SHAP values for the top-K classes.
# -------------------------------------------------------------------

    # -- Step 6: Background tensor (input excluded)  [OPT-6]
    print("\nBuilding SHAP background (input sample excluded) ...")
    background_tensor = build_background_tensor(
        mat_files, input_path, n_bg=args.n_bg
    )

    # -- Step 7: GradientExplainer
    wrapped_model = _AmpWrapper(ensemble_model)
    explainer     = shap.GradientExplainer(wrapped_model, background_tensor)
    print(f"GradientExplainer ready.")

    # -- Step 8: SHAP values for the input ECG
    print(f"\nComputing SHAP values for {record_name} "
          f"(top-{args.top_k} classes) ...")
    result = explainer.shap_values(
        sample_tensor, ranked_outputs=args.top_k
    )
    # SHAP 0.45+ with ranked_outputs returns (values, indexes).
    # values  : ndarray  (n_samples, 12, 5000, top_k)  — last axis = ranked classes
    # indexes : ndarray  (n_samples, top_k)             — class indices, NOT a tensor
    values, indexes = result
    values_arr = np.array(values)          # ensure ndarray (1, 12, 5000, top_k)
    # Rearrange to (top_k, 12, 5000) so shap_3d[rank] = (12, 5000)
    shap_3d       = np.moveaxis(values_arr[0], -1, 0)   # (top_k, 12, 5000)
    explained_idx = np.array(indexes[0]).astype(int)     # (top_k,)  — plain numpy, no .cpu()

    print(f"SHAP done. Output shape: {shap_3d.shape}  "
          f"(top_{args.top_k}, 12 leads, {SIGNAL_LEN} time steps)")
## cell 15 end


## cell 16 start
# -------------------------------------------------------------------
# Main pipeline - Steps 9-10
# For each top-K class: generate SHAP-highlighted ECG plot, run ablation
# (zero out top SHAP regions and re-score), then write the text report.
# -------------------------------------------------------------------

    # -- Step 9: Highlight + ablate for every top-K class
    print(f"\nGenerating outputs for {record_name} ...")
    ablation_results = {}

    for rank in range(len(explained_idx)):
        cls_idx  = explained_idx[rank]
        cls_name = class_names[cls_idx]
        print(f"\n[{rank+1}/{len(explained_idx)}]  {cls_name}  "
              f"p = {probs[cls_idx]:.4f}")

        # Plot 1: the actual input ECG with SHAP red/blue dots
        plot_shap_highlighted_ecg(
            sample_np, shap_3d, explained_idx,
            class_names, probs, time_axis,
            record_name=record_name,
            explained_rank=rank,
            percentile=args.percentile,
            output_dir=args.output_dir
        )

        # Plot 2 + console report: ablate top-SHAP regions, re-score
        delta, ablate_mask = run_ablation(
            sample_np, ensemble_model, shap_3d, explained_idx,
            class_names, probs, time_axis,
            record_name=record_name,
            explained_rank=rank,
            ablate_percentile=args.ablate_percentile,
            output_dir=args.output_dir
        )
        ablation_results[cls_name] = {
            "prob"        : float(probs[cls_idx]),
            "prob_ablated": float(probs[cls_idx] - delta),
            "delta"       : float(delta),
            "pass"        : bool(delta > 0.05),
            "steps_zeroed": int(ablate_mask.sum()),
        }

    # -- Step 10: Text ablation summary
    summary_path = os.path.join(
        args.output_dir, f"ablation_report_{record_name}.txt"
    )
    with open(summary_path, "w") as f:
        f.write(f"Ablation Report\n")
        f.write(f"Record    : {record_name}\n")
        f.write(f"Input file: {input_path}\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Class':<14} {'p_orig':>7} {'p_ablated':>10} "
                f"{'drop':>8}  {'zeroed':>12}  result\n")
        f.write("-" * 65 + "\n")
        for cls_name, r in ablation_results.items():
            status = "PASS" if r["pass"] else "WARN"
            f.write(
                f"{cls_name:<14} {r['prob']:>7.4f} {r['prob_ablated']:>10.4f} "
                f"{r['delta']:>+8.4f}  "
                f"{r['steps_zeroed']:>5}/{SIGNAL_LEN}  [{status}]\n"
            )
    print(f"\nAblation report: {summary_path}")

    # -- Done
    print(f"\n{'='*60}")
    print(f"  Done.  Outputs saved to: {os.path.abspath(args.output_dir)}/")
    print(f"  Files generated:")
    for cls_name in ablation_results:
        print(f"    shap_highlighted_{record_name}_{cls_name}.png")
        print(f"    shap_ablated_{record_name}_{cls_name}.png")
    print(f"    ablation_report_{record_name}.txt")
    print(f"{'='*60}\n")
## cell 16 end


## cell 17 start
# -------------------------------------------------------------------
# Entry point
# When running as a script, parse CLI args and call main().
# In a notebook, you can call main() directly after overriding
# sys.argv, or call the individual pipeline steps (cells 14-16) inline.
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
## cell 17 end