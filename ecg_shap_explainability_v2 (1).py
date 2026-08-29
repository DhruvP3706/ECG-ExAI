"""
explainability/ecg_shap_explainability_v2.py
=============================================
SHAP-based explainability + FAITHFULNESS-VALIDATED ablation for LightECGNetV2.

WHAT CHANGED FROM v1  (base approach is UNCHANGED: SHAP attribution +
ablation sanity check remain the spine).  Every change below is CLASS-AGNOSTIC:
it improves the explanation for all 45 classes equally and singles out none.

  [NEW-1] Normal-sinus-rhythm (NSR) SHAP background.
          The background is now drawn ONLY from normal ECGs (optionally
          heart-rate-matched to the input), instead of arbitrary records.
          Every SHAP value now means "deviation from a normal heart that
          pushes toward this diagnosis" — the language of differential
          diagnosis — identically for every class.

  [NEW-2] Interpolation ablation baseline (removes the masking artifact).
          Zeroing a contiguous region injects step-edges the CNN can react
          to as artifacts, confounding the confidence drop. We replace masked
          samples with within-lead linear interpolation, isolating
          "information removed" from "edge introduced". (Zero baseline is
          still selectable for comparison.)

  [NEW-3] Random + lowest-SHAP ablation CONTROLS (differential faithfulness).
          A drop is only meaningful relative to a control. For every class we
          now ablate an equal number of (a) top-|SHAP|, (b) random, and
          (c) lowest-|SHAP| samples, and PASS only if the salient drop beats
          the random drop by a margin. This makes the PASS/WARN defensible.

  [NEW-4] Deletion curves (continuous faithfulness metric + AUC).
          Instead of one quantized number, we sweep the fraction of samples
          removed and plot probability vs fraction for MoRF / random / LeRF,
          reporting the area under each curve and a faithfulness gap. This
          gives resolution the 3-fold ensemble's coarse (thirds) drop lacks.

  [NEW-5] Full determinism (seeded).
          Background sampling and random controls are seeded, so every figure
          and number in the paper is reproducible.

NOTE (intentionally NOT added, and why):
  * Per-lead energy normalization of SHAP is a no-op here: preprocessing
    already z-scores each lead to unit variance, so lead scales are already
    equalized. Adding it would be dead code.
  * Gaussian smoothing / dot-vs-band cosmetics are omitted: the outputs are
    contiguous shaded bands already; smoothing does not fix legibility.
  * Beat-level / wave-segment aggregation and RR / lead-territory panels are
    CLASS-FAMILY-specific (they single out timing vs morphology classes) and
    are therefore out of scope for this "benefits-all-classes" revision.

RUNNING IN A NOTEBOOK
---------------------
Run the cells top to bottom. Definition cells (1-15) can be run/edited in
isolation. The pipeline cells (16-19) execute at module scope and store their
intermediate results in globals (sample_np, shap_3d, ablation_results, ...),
so you can inspect each stage. In a notebook, edit the PARAMS namespace in
cell 16; as a script, pass CLI flags (argparse is used automatically).

OUTPUTS (named after the input record for traceability)
  shap_highlighted_<record>_<CLASS>.png   -- SHAP-highlighted 12-lead ECG
  shap_ablated_<record>_<CLASS>.png       -- salient ablation, per-lead bands
  deletion_curve_<record>_<CLASS>.png     -- MoRF/random/LeRF deletion curves
  ablation_report_<record>.txt            -- drops, controls, verdicts, AUCs
"""

## cell 1 start
# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import os
import sys
import csv
import glob
import random
import argparse
import warnings
from types import SimpleNamespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")          # [OPT-8] headless — no display server needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection   # [NEW-6] line-based SHAP highlight

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, find_peaks   # find_peaks: HR estimate

import torch
import torch.nn as nn
import torch.nn.functional as F

import shap

warnings.filterwarnings("ignore")
## cell 1 end


## cell 2 start
# -------------------------------------------------------------------
# Device, AMP, and GLOBAL DETERMINISM  [NEW-5]
# Seeding here makes background sampling and random ablation controls
# reproducible, so every figure/number in the paper can be regenerated.
# -------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
# -------------------------------------------------------------------
SIGNAL_LEN  = 5000
FS          = 500
NUM_CLASSES = 45

# --- Checkpoints (one .pt per CV fold) ---
MODEL_PATHS = [
    r"C:\Users\lenovo\HonsExAI\lightv2_ft_fold0.pt",
    r"C:\Users\lenovo\HonsExAI\lightv2_p1_fold1.pt",
    r"C:\Users\lenovo\HonsExAI\lightv2_p1_fold2.pt",
]

# --- Dataset + condition-name mapping ---
DATASET_ROOT = r"C:\Users\lenovo\HonsExAI\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords"
MAPPING_CSV  = r"C:\Users\lenovo\HonsExAI\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv"

# ===================================================================
# [NEW-1] Normal-sinus background configuration
# NORMAL_ACRONYMS must match your label vocabulary. In the Chapman/Ningbo
# SNOMED mapping, sinus rhythm is acronym "SR" (SNOMED 426783006). Verify
# against ConditionNames_SNOMED-CT.csv. Brady/tachy ("SB"/"ST") are NOT
# normal and are deliberately excluded.
# ===================================================================
NORMAL_ACRONYMS          = {"SR"}      # <-- set to your normal-sinus acronym(s)
STRICT_NORMAL_BACKGROUND = False       # True: error if too few NSR; False: warn + top up
HR_MATCH_BACKGROUND      = True        # match background HR to the input's HR
HR_TOLERANCE_BPM         = 15.0        # +/- window for HR matching

# ===================================================================
# [NEW-2] Ablation baseline mode
#   "interp" (recommended) : within-lead linear interpolation (no step edges)
#   "zero"                 : original z-score-mean zeroing (kept for comparison)
# ===================================================================
BASELINE_MODE = "interp"

# ===================================================================
# [NEW-3] Differential-ablation margin: the salient (top-|SHAP|) drop must
# exceed the random-control drop by at least this much to count as PASS.
# ===================================================================
ABLATION_MARGIN = 0.05

# ===================================================================
# [NEW-4] Deletion-curve sampling (fraction of samples removed)
# ===================================================================
PERTURB_FRACTIONS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

# Quick sanity-check of resolved paths.
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
# Preprocessing  [OPT-1]  + heart-rate estimate (for [NEW-1] HR matching)
# Preprocessing MUST match training exactly.
# -------------------------------------------------------------------
_BP_SOS = butter(3, [0.5 / (0.5 * FS), 40.0 / (0.5 * FS)],
                 btype="band", output="sos")


def preprocess_ecg_mat(mat_path, target_len=SIGNAL_LEN):
    """
    Load a PhysioNet .mat ECG and apply training-identical preprocessing:
      1. read 'val'/'data'  2. ensure (12, T)  3. pad/crop  4. bandpass
      5. per-lead z-score.  Returns np.ndarray (12, target_len) float32.
    NOTE: per-lead z-scoring here is why no extra per-lead energy
    normalization of SHAP is needed later — lead scales are already equal.
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
            [signal, np.zeros((12, target_len - cur_len), dtype=signal.dtype)], axis=1)
    elif cur_len > target_len:
        signal = signal[:, :target_len]

    signal = sosfiltfilt(_BP_SOS, signal.astype(np.float32), axis=-1)
    mean   = signal.mean(axis=1, keepdims=True)
    std    = signal.std(axis=1,  keepdims=True) + 1e-8
    return ((signal - mean) / std).astype(np.float32)


def estimate_hr(signal, fs=FS, lead=1):
    """
    Rough heart rate (bpm) from R-peak spacing on one lead (default II).
    Used only to HR-match the NSR background. Returns np.nan on failure.
    Signal is z-scored, so prominence is in standard-deviation units.
    """
    for l in (lead, 0, 6):                       # try II, then I, then V1
        x = signal[l]
        peaks, _ = find_peaks(x, distance=int(0.3 * fs), prominence=1.0)
        if len(peaks) >= 2:
            rr = np.diff(peaks) / fs
            rr = rr[(rr > 0.3) & (rr < 2.0)]     # 30-200 bpm plausibility
            if len(rr) > 0:
                return float(60.0 / np.median(rr))
    return float("nan")
## cell 4 end


## cell 5 start
# -------------------------------------------------------------------
# SNOMED map + per-record label reader  (supports [NEW-1] NSR filtering)
# -------------------------------------------------------------------
def load_snomed_map(mapping_csv):
    """SNOMED-CT code -> acronym name."""
    m = {}
    with open(mapping_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m[row["Snomed_CT"].strip()] = row["Acronym Name"].strip()
    return m


def record_acronyms(mat_path, snomed_map):
    """Read the record's .hea #Dx line and return its set of acronym labels."""
    hea = os.path.splitext(mat_path)[0] + ".hea"
    codes = []
    try:
        with open(hea) as f:
            for line in f:
                if "#Dx" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        codes = [c.strip() for c in parts[1].split(",") if c.strip()]
                    break
    except Exception:
        return set()
    return {snomed_map[c] for c in codes if c in snomed_map}
## cell 5 end


## cell 6 start
# -------------------------------------------------------------------
# Model building blocks (unchanged — keep in sync with models/lightecgnetv2.py)
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
        self.dw  = nn.Conv1d(in_ch, in_ch,  kernel, padding=pad, groups=in_ch, bias=False)
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
        self.dw   = nn.Conv1d(in_ch, in_ch, kernel, padding=pad, dilation=dilation,
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
## cell 6 end


## cell 7 start
# -------------------------------------------------------------------
# LightECGNetV2 (unchanged)
# -------------------------------------------------------------------
class LightECGNetV2(nn.Module):
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
## cell 7 end


## cell 8 start
# -------------------------------------------------------------------
# EnsembleModel + load_ensemble + AMP wrapper (unchanged)
# -------------------------------------------------------------------
class EnsembleModel(nn.Module):
    """Averages sigmoid probabilities from the fold checkpoints (multi-label)."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        probs = [torch.sigmoid(m(x)) for m in self.models]
        return torch.stack(probs, dim=0).mean(dim=0)


def load_ensemble(model_paths):
    models = []
    for path in model_paths:
        print(f"  Loading: {path}")
        m = LightECGNetV2(num_classes=NUM_CLASSES).to(device)
        # weights_only=False: checkpoints are OrderedDicts saved without SafeTensor format
        state = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)

    ensemble = EnsembleModel(models).to(device)
    ensemble.eval()

    if int(torch.__version__.split(".")[0]) >= 2 and device.type == "cuda":
        try:
            ensemble = torch.compile(ensemble)
            print("  [OPT-3] torch.compile() applied.")
        except Exception as e:
            print(f"  [OPT-3] torch.compile() skipped: {e}")

    print(f"Ensemble ready — {len(models)} fold models.")
    return ensemble


class _AmpWrapper(nn.Module):
    """Wraps ensemble in autocast [OPT-5]; returns float32 for SHAP."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with _amp_ctx:
            return self.model(x).float()
## cell 8 end


## cell 9 start
# -------------------------------------------------------------------
# Class name loader (refactored to reuse load_snomed_map)
# -------------------------------------------------------------------
def load_class_names(mapping_csv, dataset_root, exclude=None):
    cache = "class_names.npy"
    if os.path.exists(cache):
        names = np.load(cache, allow_pickle=True).tolist()
        print(f"Loaded class_names from cache ({len(names)} classes).")
        return names

    if exclude is None:
        exclude = {'ABI', 'VET', 'FQRS', 'SAAWR', 'JPT', 'VB'}

    snomed_to_acronym = load_snomed_map(mapping_csv)

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
## cell 9 end


## cell 10 start
# -------------------------------------------------------------------
# [NEW-1] Normal-sinus-rhythm background (optionally HR-matched)
# -------------------------------------------------------------------
def build_background_tensor(mat_files, input_path, snomed_map, input_hr=None,
                            n_bg=8, max_workers=8, rng=None):
    """
    Build a SHAP background from NORMAL records only, so attributions read as
    "deviation from normal that drives this diagnosis" for every class.

      [NEW-1] filter candidates to NORMAL_ACRONYMS (+ optional HR match)
      [OPT-6] exclude the input file from its own background
      [OPT-2] parallel loading
      [NEW-5] deterministic candidate order (seeded)

    Returns torch.Tensor (n_bg, 12, SIGNAL_LEN) on device.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    input_abs  = os.path.abspath(input_path)
    candidates = [p for p in mat_files if os.path.abspath(p) != input_abs]   # [OPT-6]
    order      = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]                              # deterministic

    # --- gather a shortlist of NORMAL records ---
    normal = []
    for p in candidates:
        if len(normal) >= n_bg * 4:
            break
        if NORMAL_ACRONYMS & record_acronyms(p, snomed_map):
            normal.append(p)

    # --- optional HR match on the normal shortlist ---
    shortlist = normal
    if HR_MATCH_BACKGROUND and input_hr is not None and not np.isnan(input_hr):
        hr_ok = []
        for p in normal:
            try:
                hr = estimate_hr(preprocess_ecg_mat(p))
                if not np.isnan(hr) and abs(hr - input_hr) <= HR_TOLERANCE_BPM:
                    hr_ok.append(p)
            except Exception:
                continue
            if len(hr_ok) >= n_bg:
                break
        if len(hr_ok) >= max(1, n_bg // 2):
            shortlist = hr_ok
            print(f"  [bg] HR-matched normals: {len(hr_ok)} within "
                  f"+/-{HR_TOLERANCE_BPM:.0f} bpm of input ({input_hr:.0f} bpm).")
        else:
            print("  [bg] Too few HR-matched normals; using normals without HR match.")

    chosen = shortlist[:n_bg]
    used_fallback = False

    # --- handle shortfall ---
    if len(chosen) < n_bg:
        if STRICT_NORMAL_BACKGROUND:
            raise RuntimeError(
                f"Only {len(chosen)} normal-sinus background records found; need {n_bg}. "
                f"Set STRICT_NORMAL_BACKGROUND=False to top up with other records, "
                f"or check NORMAL_ACRONYMS={NORMAL_ACRONYMS}.")
        used_fallback = True
        for p in candidates:
            if len(chosen) >= n_bg:
                break
            if p not in chosen:
                chosen.append(p)
        print(f"  [bg] WARNING: only {len(shortlist)} normal record(s); topped up to "
              f"{len(chosen)} with non-normal records (baseline not purely normal).")

    # --- parallel load [OPT-2] ---
    bg_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(preprocess_ecg_mat, p): p for p in chosen}
        for fut in as_completed(futures):
            try:
                bg_list.append(fut.result())
            except Exception as e:
                print(f"  Skipping {futures[fut]}: {e}")

    if not bg_list:
        raise RuntimeError("Could not load any background samples.")

    bg = torch.tensor(np.stack(bg_list), dtype=torch.float32).to(device)
    tag = "normal-sinus" + ("" if not used_fallback else " + fallback")
    print(f"Background: {tuple(bg.shape)}  ({tag}; input excluded [OPT-6])")
    return bg
## cell 10 end


## cell 11 start
# -------------------------------------------------------------------
# Scoring, ablation baselines, and mask builders
#   [NEW-2] interp baseline (no step-edge artifact)  vs  zero baseline
#   [NEW-3] equal-count top / random / lowest masks (controls)
# -------------------------------------------------------------------
def score_prob(sample_np, cls_idx, ensemble_model):
    """Ensemble probability for one class on a (12, 5000) array. [OPT-4, OPT-5]"""
    x = torch.tensor(np.asarray(sample_np, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode(), _amp_ctx:
        return float(ensemble_model(x)[0, cls_idx].float().item())


def zero_baseline(sample, mask):
    """Original ablation: set masked samples to 0.0 (= z-score mean)."""
    out = sample.copy()
    out[mask] = 0.0
    return out.astype(np.float32)


def interp_baseline(sample, mask):
    """
    [NEW-2] Replace masked samples with WITHIN-LEAD linear interpolation.
    Removes the information at those samples without injecting the step-edges
    that zeroing a contiguous region would, so the confidence drop reflects
    lost signal rather than a masking artifact.
    """
    out = sample.copy()
    T = sample.shape[1]
    t = np.arange(T)
    for lead in range(sample.shape[0]):
        m = mask[lead]
        if m.any() and (~m).any():
            out[lead, m] = np.interp(t[m], t[~m], sample[lead, ~m])
        elif m.all():
            out[lead, :] = 0.0            # degenerate: nothing to interpolate from
    return out.astype(np.float32)


def get_baseline_fn(mode=None):
    mode = mode or BASELINE_MODE
    return interp_baseline if mode == "interp" else zero_baseline


def baseline_label(mode=None):
    mode = mode or BASELINE_MODE
    return "interpolated" if mode == "interp" else "zeroed to 0"


def build_masks(sv, percentile, rng):
    """
    [NEW-3] Per-lead, equal-count masks for the three ablation conditions:
        m_top : the top-(100-percentile)% |SHAP| samples  (salient)
        m_rnd : an equal number of RANDOM samples          (control)
        m_low : an equal number of the LOWEST |SHAP| samples (control)
    Equal counts per lead make the three conditions directly comparable.
    """
    L, T = sv.shape
    m_top = np.zeros((L, T), dtype=bool)
    m_rnd = np.zeros((L, T), dtype=bool)
    m_low = np.zeros((L, T), dtype=bool)
    for l in range(L):
        mag = np.abs(sv[l])
        thr = np.percentile(mag, percentile)
        k   = int(np.count_nonzero(mag >= thr))     # count defines all three
        if k <= 0:
            continue
        order = np.argsort(mag)                      # ascending
        m_top[l, order[-k:]] = True                 # largest |SHAP|
        m_low[l, order[:k]]  = True                 # smallest |SHAP|
        m_rnd[l, rng.choice(T, size=min(k, T), replace=False)] = True
    return m_top, m_rnd, m_low
## cell 11 end


## cell 12 start
# -------------------------------------------------------------------
# Plot 1: SHAP-highlighted input ECG
#
#   [NEW-6] LINE-BASED HIGHLIGHTING (replaces dot-scatter overlay).
#   The salient stretches of the waveform are now drawn as a THICKER,
#   COLORED SEGMENT OF THE ECG LINE ITSELF -- red where SHAP is positive
#   (supports the class), blue where negative (opposes it) -- instead of
#   dots sitting on top of the trace. Non-salient stretches remain a thin
#   grey/black line for context. This mirrors the ablation plot's
#   contiguous grey bands: a clinician now sees "this stretch of the beat"
#   rather than a scatter of unrelated points, which is easier to read and
#   easier to compare directly against named ECG segments (P/QRS/T, etc.).
#   This is a pure VISUALIZATION change -- the underlying SHAP values,
#   background, and thresholding are all unchanged.
# -------------------------------------------------------------------
def _salient_line_segments(time_axis, y, s, percentile):
    """
    Build (segments, colors) for a LineCollection that highlights the
    waveform itself wherever |SHAP| clears the percentile threshold.

    A segment (the line between sample i and i+1) is drawn in color only if
    EITHER endpoint is salient, so highlighted stretches stay visually
    contiguous rather than breaking into single-sample slivers. Its color
    (red/blue) follows whichever endpoint has the larger |SHAP|, matching
    the dominant polarity of that stretch.
    """
    thr = np.percentile(np.abs(s), percentile)
    salient = np.abs(s) >= thr

    pts  = np.column_stack([time_axis, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)          # (N-1, 2, 2)

    seg_salient = salient[:-1] | salient[1:]
    # polarity = sign of whichever endpoint carries the larger |SHAP|
    dominant_is_left = np.abs(s[:-1]) >= np.abs(s[1:])
    seg_sign = np.where(dominant_is_left, s[:-1], s[1:])

    colors = np.zeros((len(segs), 4))                            # RGBA, default transparent
    colors[seg_salient & (seg_sign > 0)] = (0.80, 0.10, 0.10, 1.0)   # red  = positive SHAP
    colors[seg_salient & (seg_sign <= 0)] = (0.12, 0.35, 0.75, 1.0)  # blue = negative SHAP
    return segs, colors


def plot_shap_highlighted_ecg(sample_np, shap_3d, explained_idx,
                               class_names, probs, time_axis,
                               record_name, explained_rank=0,
                               percentile=90, output_dir="."):
    """
    Plot all 12 leads of the INPUT ECG with SHAP attribution highlighted
    directly ON the waveform (line-based, not dots).

      Red  stretch = positive SHAP (time regions that push the model TOWARD
                      predicting this class for this patient's ECG)
      Blue stretch = negative SHAP (time regions that push the model AWAY
                      from predicting this class for this patient's ECG)
      Thin grey/black = non-salient waveform, shown for anatomical context.

    Only the top (100-percentile)% magnitude SHAP values are highlighted,
    same thresholding as before -- only the rendering changed.

    Output: shap_highlighted_<record>_<CLASS>.png
    """
    sv       = shap_3d[explained_rank]   # (12, 5000) for this class
    cls_idx  = explained_idx[explained_rank]
    cls_name = class_names[cls_idx]
    prob_val = probs[cls_idx]

    fig, axes = plt.subplots(12, 1, figsize=(18, 22), sharex=True)

    for i, ax in enumerate(axes):
        y = sample_np[i]          # actual input ECG lead i
        s = sv[i]                 # SHAP values for this lead

        # Full waveform, thin and muted -- provides anatomical context for
        # the highlighted stretches and keeps non-salient morphology visible.
        ax.plot(time_axis, y, color="black", lw=0.7, alpha=0.45, zorder=1)

        # Salient stretches, drawn as a thicker colored segment of the line
        # itself (not a dot overlay) -- [NEW-6].
        segs, colors = _salient_line_segments(time_axis, y, s, percentile)
        lc = LineCollection(segs, colors=colors, linewidths=2.2, zorder=3,
                            capstyle="round", joinstyle="round")
        ax.add_collection(lc)

        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=22, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_xlim(0, SIGNAL_LEN / FS)
        # LineCollection does not auto-scale y; set limits from the waveform.
        pad = 0.08 * (np.ptp(y) + 1e-6)
        ax.set_ylim(y.min() - pad, y.max() + pad)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    legend_items = [
        Line2D([0], [0], color="black", lw=1.0, alpha=0.45, label="ECG waveform (non-salient)"),
        Line2D([0], [0], color=(0.80, 0.10, 0.10), lw=2.4,
               label=f"Positive SHAP — supports '{cls_name}' (top {100-percentile:.0f}% |SHAP|)"),
        Line2D([0], [0], color=(0.12, 0.35, 0.75), lw=2.4,
               label=f"Negative SHAP — opposes '{cls_name}' (top {100-percentile:.0f}% |SHAP|)"),
    ]
    fig.legend(handles=legend_items, loc="upper right",
               bbox_to_anchor=(0.98, 0.99), frameon=True, fontsize=9)
    fig.suptitle(
        f"SHAP-Highlighted ECG (normal-sinus background)  —  {record_name}\n"
        f"Predicted: {cls_name}   p = {prob_val:.4f}   "
        f"(rank #{explained_rank + 1} of top-{len(explained_idx)})",
        fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 0.97, 0.990])

    out_path = os.path.join(output_dir, f"shap_highlighted_{record_name}_{cls_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path
## cell 12 end


## cell 13 start
# -------------------------------------------------------------------
# Plot 2: Ablated input ECG (salient regions), now annotated with the
# random/lowest CONTROL drops and the differential verdict  [NEW-3]
# -------------------------------------------------------------------
def plot_ablated_ecg(sample_np, ablate_mask_2d, cls_name,
                     prob_before, prob_after, d_random, d_low,
                     baseline_txt, time_axis, record_name, passed, output_dir="."):
    delta     = prob_before - prob_after
    pass_fail = "PASS \u2713" if passed else "WARN \u2717"

    fig, axes = plt.subplots(12, 1, figsize=(18, 22), sharex=True)
    for i, ax in enumerate(axes):
        y    = sample_np[i]
        mask = ablate_mask_2d[i]
        ax.plot(time_axis, y, color="black", lw=0.9, zorder=2)
        in_region, start_t = False, 0
        for t in range(len(mask)):
            if mask[t] and not in_region:
                start_t, in_region = t, True
            elif not mask[t] and in_region:
                ax.axvspan(time_axis[start_t], time_axis[t], alpha=0.45,
                           color="grey", lw=0, zorder=1)
                in_region = False
        if in_region:
            ax.axvspan(time_axis[start_t], time_axis[-1], alpha=0.45,
                       color="grey", lw=0, zorder=1)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=22, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_xlim(0, SIGNAL_LEN / FS)
    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    legend_items = [
        Line2D([0], [0], color="black", lw=1.2, label="ECG waveform"),
        mpatches.Patch(facecolor="grey", alpha=0.45,
                       label=f"Ablated region (per-lead top 10% |SHAP| — {baseline_txt})"),
    ]
    fig.legend(handles=legend_items, loc="upper right",
               bbox_to_anchor=(0.98, 0.99), frameon=True, fontsize=9)
    fig.suptitle(
        f"Ablation Check  —  {record_name}  —  Class: {cls_name}\n"
        f"p: {prob_before:.4f} -> {prob_after:.4f}   "
        f"salient drop: {delta:+.4f}   |   controls  random: {d_random:+.4f}   "
        f"lowest: {d_low:+.4f}    {pass_fail}",
        fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 0.97, 0.990])

    out_path = os.path.join(output_dir, f"shap_ablated_{record_name}_{cls_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path
## cell 13 end


## cell 14 start
# -------------------------------------------------------------------
# Ablation logic with CONTROLS  [NEW-2, NEW-3]
# Zeroes/interpolates the salient region AND equal-count random & lowest
# regions, re-scores each, and PASSES only if the salient drop beats the
# random-control drop by ABLATION_MARGIN.
# -------------------------------------------------------------------
def run_ablation(sample_np, ensemble_model, shap_3d, explained_idx,
                 class_names, probs, time_axis, record_name, rng,
                 explained_rank=0, ablate_percentile=90, output_dir="."):
    cls_idx   = explained_idx[explained_rank]
    cls_name  = class_names[cls_idx]
    orig_prob = probs[cls_idx]
    sv        = shap_3d[explained_rank]                         # (12, 5000)

    baseline_fn = get_baseline_fn()
    b_txt       = baseline_label()

    # equal-count masks for the three conditions
    m_top, m_rnd, m_low = build_masks(sv, ablate_percentile, rng)

    def drop_for(mask):
        ablated = baseline_fn(sample_np, mask)
        return orig_prob - score_prob(ablated, cls_idx, ensemble_model)

    d_top = drop_for(m_top)
    d_rnd = drop_for(m_rnd)
    d_low = drop_for(m_low)

    # differential verdict: salient must beat the random control
    passed = (d_top > d_rnd + ABLATION_MARGIN) and (d_top > 0)
    n_cells = int(m_top.sum())

    print(f"\n  {'='*58}")
    print(f"  Ablation  --  {cls_name}   (baseline: {b_txt})")
    print(f"    Original probability     : {orig_prob:.4f}")
    print(f"    Salient  drop (top |SHAP|): {d_top:+.4f}")
    print(f"    Random   drop (control)   : {d_rnd:+.4f}")
    print(f"    Lowest   drop (control)   : {d_low:+.4f}")
    print(f"    Salient - random          : {d_top - d_rnd:+.4f}   (margin {ABLATION_MARGIN})")
    print(f"    Cells zeroed/interpolated : {n_cells} / {12 * SIGNAL_LEN} "
          f"(per-lead top {100-ablate_percentile:.0f}%)")
    if passed:
        print(f"    PASS -- salient regions are causally relevant vs. control.")
    else:
        print(f"    WARN -- salient drop does not clearly beat the random control.")
    print(f"  {'='*58}")

    plot_ablated_ecg(sample_np, m_top, cls_name, orig_prob, orig_prob - d_top,
                     d_rnd, d_low, b_txt, time_axis, record_name, passed,
                     output_dir=output_dir)

    return {"d_top": float(d_top), "d_random": float(d_rnd), "d_low": float(d_low),
            "passed": bool(passed), "cells": n_cells, "mask": m_top}
## cell 14 end


## cell 15 start
# -------------------------------------------------------------------
# Deletion curves  [NEW-4]
# Sweep the fraction of samples removed and score the model, for
#   MoRF  (most-relevant-first)  — should drop fastest if faithful
#   random                       — control
#   LeRF  (least-relevant-first) — should drop slowest
# The area under each curve (lower = confidence lost faster) and the
# faithfulness gap (random AUC - MoRF AUC) quantify attribution quality.
# -------------------------------------------------------------------
def deletion_curve(sample_np, sv, cls_idx, ensemble_model, fractions,
                   baseline_fn, rng, order="morf"):
    mag  = np.abs(sv).ravel()
    n    = mag.size
    rank = np.argsort(mag)[::-1]                 # most-relevant-first
    if order == "random":
        rank = rng.permutation(n)
    elif order == "lerf":
        rank = rank[::-1]

    probs = []
    for f in fractions:
        k = int(round(f * n))
        mask = np.zeros(n, dtype=bool)
        if k > 0:
            mask[rank[:k]] = True
        ablated = baseline_fn(sample_np, mask.reshape(sv.shape))
        probs.append(score_prob(ablated, cls_idx, ensemble_model))
    return np.asarray(probs)


def curve_auc(fractions, probs):
    fractions = np.asarray(fractions, dtype=float)
    probs     = np.asarray(probs, dtype=float)
    span = fractions[-1] - fractions[0]
    if span <= 0:
        return float("nan")
    # np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trap(probs, fractions) / span)


def plot_deletion_curves(fractions, curves, aucs, cls_name, record_name, output_dir="."):
    style = {
        "morf":   ("#c1352c", "-",  "o", "MoRF — remove most-relevant first"),
        "random": ("#5b6b73", "--", "s", "Random — control"),
        "lerf":   ("#2a6f97", ":",  "^", "LeRF — remove least-relevant first"),
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    xf = np.asarray(fractions) * 100.0
    for o in ["morf", "random", "lerf"]:
        c, ls, mk, lbl = style[o]
        ax.plot(xf, curves[o], color=c, linestyle=ls, marker=mk, ms=4, lw=1.8,
                label=f"{lbl}   (AUC={aucs[o]:.3f})")
    gap = aucs["random"] - aucs["morf"]
    ax.set_xlabel("Percent of samples ablated (%)", fontsize=11)
    ax.set_ylabel(f"Model probability for '{cls_name}'", fontsize=11)
    ax.set_title(f"Deletion Curves  —  {record_name}  —  {cls_name}\n"
                 f"Faithfulness gap (random - MoRF AUC) = {gap:+.3f}  "
                 f"({'faithful' if gap > 0 else 'weak'})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"deletion_curve_{record_name}_{cls_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def deletion_faithfulness(sample_np, sv, cls_idx, ensemble_model, rng,
                          fractions=None, baseline_fn=None):
    """Compute the three deletion curves + their AUCs for one class."""
    fractions   = PERTURB_FRACTIONS if fractions is None else fractions
    baseline_fn = get_baseline_fn() if baseline_fn is None else baseline_fn
    curves = {o: deletion_curve(sample_np, sv, cls_idx, ensemble_model,
                                fractions, baseline_fn, rng, order=o)
              for o in ["morf", "random", "lerf"]}
    aucs = {o: curve_auc(fractions, curves[o]) for o in curves}
    return curves, aucs
## cell 15 end


## cell 16 start
# -------------------------------------------------------------------
# Parameters (notebook vs. script)
# In a NOTEBOOK: edit the PARAMS fields below, then run the pipeline cells.
# As a SCRIPT  : CLI flags are parsed automatically.
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP explainability + faithfulness-validated ablation for LightECGNetV2")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to the .mat ECG to explain.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top predicted classes to explain.")
    parser.add_argument("--percentile", type=float, default=90.0,
                        help="SHAP threshold percentile for dot highlighting.")
    parser.add_argument("--ablate_percentile", type=float, default=90.0,
                        help="Percentile above which samples are ablated.")
    parser.add_argument("--output_dir", "-o", type=str, default=".",
                        help="Directory to save outputs.")
    parser.add_argument("--n_bg", type=int, default=8,
                        help="Number of NORMAL background samples for SHAP.")
    return parser.parse_args()


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


if _in_notebook():
    # ---- EDIT THESE IN THE NOTEBOOK ----
    PARAMS = SimpleNamespace(
        input=None,                 # e.g. r"...\WFDBRecords\01\010\JS00001.mat"
        top_k=5,
        percentile=90.0,
        ablate_percentile=90.0,
        output_dir=".",
        n_bg=8,
    )
    print("Notebook mode: edit PARAMS in cell 16 as needed, then run cells 17-19.")
else:
    PARAMS = parse_args()

os.makedirs(PARAMS.output_dir, exist_ok=True)

# Shared RNG for background selection + random ablation controls  [NEW-5]
RNG = np.random.default_rng(SEED)
## cell 16 end


## cell 17 start
# -------------------------------------------------------------------
# PIPELINE  Steps 1-5 : resolve input, preprocess, load labels + ensemble,
# score the ECG.  (Runs at module scope; results stored in globals.)
# -------------------------------------------------------------------
mat_files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.mat"), recursive=True)
if not mat_files:
    raise FileNotFoundError(
        f"No .mat files found under {DATASET_ROOT}. Check DATASET_ROOT (cell 3).")

# -- Step 1: resolve input path
if PARAMS.input is not None:
    input_path = PARAMS.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
else:
    input_path = mat_files[0]
    print("\n  WARNING: No --input / PARAMS.input specified.\n"
          f"  Falling back to: {input_path}\n"
          "  For meaningful results, specify your target ECG.\n")

record_name = os.path.splitext(os.path.basename(input_path))[0]
print(f"\n{'='*60}\n  Input ECG : {input_path}\n  Record    : {record_name}\n{'='*60}\n")

# -- Step 2: preprocess input + estimate HR (for NSR background matching)
print("Preprocessing input ECG ...")
sample_np     = preprocess_ecg_mat(input_path)                      # (12, 5000)
sample_tensor = torch.tensor(sample_np).unsqueeze(0).to(device)     # (1, 12, 5000)
time_axis     = np.arange(SIGNAL_LEN) / FS
input_hr      = estimate_hr(sample_np)
print(f"  Estimated input HR: {input_hr:.0f} bpm" if not np.isnan(input_hr)
      else "  Estimated input HR: n/a")

# -- Step 3: labels + SNOMED map (map reused for NSR background filtering)
snomed_map  = load_snomed_map(MAPPING_CSV)
class_names = load_class_names(MAPPING_CSV, DATASET_ROOT)

# -- Step 4: ensemble (+ warm-up for torch.compile)
print("\nLoading ensemble ...")
ensemble_model = load_ensemble(MODEL_PATHS)
_dummy = torch.randn(1, 12, SIGNAL_LEN).to(device)
with torch.inference_mode(), _amp_ctx:
    ensemble_model(_dummy)
del _dummy

# -- Step 5: score input
print(f"\nScoring {record_name} ...")
with torch.inference_mode(), _amp_ctx:
    probs = ensemble_model(sample_tensor)[0].float().cpu().numpy()  # (45,)

top_k_idx = np.argsort(probs)[::-1][:PARAMS.top_k]
print(f"\nTop {PARAMS.top_k} predictions for {record_name}:")
for rank, idx in enumerate(top_k_idx, 1):
    print(f"  {rank}. {class_names[idx]:10s}  p = {probs[idx]:.4f}")
## cell 17 end


## cell 18 start
# -------------------------------------------------------------------
# PIPELINE  Steps 6-8 : NSR background [NEW-1], GradientExplainer, SHAP values.
# -------------------------------------------------------------------
# -- Step 6: normal-sinus background (input excluded, optionally HR-matched)
print("\nBuilding SHAP background (normal-sinus; input excluded) ...")
background_tensor = build_background_tensor(
    mat_files, input_path, snomed_map, input_hr=input_hr,
    n_bg=PARAMS.n_bg, rng=RNG)

# -- Step 7: GradientExplainer (base approach UNCHANGED)
wrapped_model = _AmpWrapper(ensemble_model)
explainer     = shap.GradientExplainer(wrapped_model, background_tensor)
print("GradientExplainer ready.")

# -- Step 8: SHAP values for the input, top-K classes
print(f"\nComputing SHAP values for {record_name} (top-{PARAMS.top_k}) ...")
result = explainer.shap_values(sample_tensor, ranked_outputs=PARAMS.top_k)
values, indexes = result
values_arr    = np.array(values)                       # (1, 12, 5000, top_k)
shap_3d       = np.moveaxis(values_arr[0], -1, 0)      # (top_k, 12, 5000)
explained_idx = np.array(indexes[0]).astype(int)       # (top_k,)
print(f"SHAP done. Output shape: {shap_3d.shape}  "
      f"(top_{PARAMS.top_k}, 12 leads, {SIGNAL_LEN} steps)")
## cell 18 end


## cell 19 start
# -------------------------------------------------------------------
# PIPELINE  Steps 9-11 : per class -> highlight, ablation + controls [NEW-3],
# deletion curves [NEW-4]; then write the text report.
# -------------------------------------------------------------------
print(f"\nGenerating outputs for {record_name} ...")
ablation_results = {}
baseline_fn      = get_baseline_fn()

for rank in range(len(explained_idx)):
    cls_idx  = explained_idx[rank]
    cls_name = class_names[cls_idx]
    print(f"\n[{rank+1}/{len(explained_idx)}]  {cls_name}  p = {probs[cls_idx]:.4f}")

    # Plot 1: SHAP-highlighted ECG (normal-sinus background)
    plot_shap_highlighted_ecg(
        sample_np, shap_3d, explained_idx, class_names, probs, time_axis,
        record_name=record_name, explained_rank=rank,
        percentile=PARAMS.percentile, output_dir=PARAMS.output_dir)

    # Plot 2 + console: ablation with random/lowest controls
    abl = run_ablation(
        sample_np, ensemble_model, shap_3d, explained_idx, class_names, probs,
        time_axis, record_name=record_name, rng=RNG, explained_rank=rank,
        ablate_percentile=PARAMS.ablate_percentile, output_dir=PARAMS.output_dir)

    # Plot 3 + metrics: deletion curves (MoRF / random / LeRF) + AUCs
    curves, aucs = deletion_faithfulness(
        sample_np, shap_3d[rank], cls_idx, ensemble_model, RNG,
        fractions=PERTURB_FRACTIONS, baseline_fn=baseline_fn)
    plot_deletion_curves(PERTURB_FRACTIONS, curves, aucs, cls_name,
                         record_name, output_dir=PARAMS.output_dir)
    print(f"    Deletion AUC  MoRF={aucs['morf']:.3f}  "
          f"random={aucs['random']:.3f}  LeRF={aucs['lerf']:.3f}  "
          f"(gap {aucs['random']-aucs['morf']:+.3f})")

    ablation_results[cls_name] = {
        "prob":          float(probs[cls_idx]),
        "d_top":         abl["d_top"],
        "d_random":      abl["d_random"],
        "d_low":         abl["d_low"],
        "passed":        abl["passed"],
        "cells":         abl["cells"],
        "auc_morf":      aucs["morf"],
        "auc_random":    aucs["random"],
        "auc_lerf":      aucs["lerf"],
        "faith_gap":     float(aucs["random"] - aucs["morf"]),
    }

# -- Step 11: text report
summary_path = os.path.join(PARAMS.output_dir, f"ablation_report_{record_name}.txt")
with open(summary_path, "w") as f:
    f.write("Ablation & Faithfulness Report\n")
    f.write(f"Record       : {record_name}\n")
    f.write(f"Input file   : {input_path}\n")
    f.write(f"Background   : normal-sinus (acronyms={sorted(NORMAL_ACRONYMS)}), "
            f"HR-match={HR_MATCH_BACKGROUND} (+/-{HR_TOLERANCE_BPM:.0f} bpm), "
            f"input HR={input_hr:.0f} bpm\n" if not np.isnan(input_hr) else
            f"Background   : normal-sinus (acronyms={sorted(NORMAL_ACRONYMS)}), "
            f"HR-match={HR_MATCH_BACKGROUND}\n")
    f.write(f"Ablation     : baseline={BASELINE_MODE} ({baseline_label()}), "
            f"controls=random+lowest, PASS if salient-random>{ABLATION_MARGIN}\n")
    f.write("=" * 92 + "\n")
    f.write(f"{'Class':<12} {'p':>6} {'salient':>8} {'random':>8} {'lowest':>8}  "
            f"{'AUC_MoRF':>8} {'AUC_rnd':>8} {'gap':>7}  result\n")
    f.write("-" * 92 + "\n")
    for cls_name, r in ablation_results.items():
        status = "PASS" if r["passed"] else "WARN"
        f.write(f"{cls_name:<12} {r['prob']:>6.3f} {r['d_top']:>+8.4f} "
                f"{r['d_random']:>+8.4f} {r['d_low']:>+8.4f}  "
                f"{r['auc_morf']:>8.3f} {r['auc_random']:>8.3f} "
                f"{r['faith_gap']:>+7.3f}  [{status}]\n")
    f.write("-" * 92 + "\n")
    f.write("salient/random/lowest = probability drop when that region is ablated.\n")
    f.write("AUC_MoRF < AUC_rnd (positive gap) indicates a faithful attribution.\n")
print(f"\nAblation report: {summary_path}")

print(f"\n{'='*60}")
print(f"  Done.  Outputs in: {os.path.abspath(PARAMS.output_dir)}/")
for cls_name in ablation_results:
    print(f"    shap_highlighted_{record_name}_{cls_name}.png")
    print(f"    shap_ablated_{record_name}_{cls_name}.png")
    print(f"    deletion_curve_{record_name}_{cls_name}.png")
print(f"    ablation_report_{record_name}.txt")
print(f"{'='*60}\n")
## cell 19 end
