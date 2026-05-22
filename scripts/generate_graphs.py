"""
Generate all graphs for the tt_detector presentation.
Focuses on the signal processing pipeline and the pre-peak mechanism.
"""

import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt
from pathlib import Path
import sys

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
RAW_DIR = BASE / "data" / "raw_sounds"
CSV_PATH = BASE / "data" / "full.csv"
SOUNDS_DIR = BASE / "data" / "sounds"

# - tt_sounds/src: for original paper Detector class
# - BASE/src: so `import audio_utils` resolves inside tt_sounds
# - BASE: so `import src.audio_utils` resolves inside project detectors
for p in [str(BASE / "tt_sounds" / "src"), str(BASE / "src"), str(BASE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from audio_utils import open_audio, highpass_filter

OUT = BASE / "presentation" / "public" / "imgs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "text.color":       "#c9d1d9",
    "figure.titlesize": 16,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
}
plt.rcParams.update(STYLE)

C_RAW        = "#58a6ff"   # blue
C_FILTERED   = "#3fb950"   # green
C_ENERGY     = "#d2a8ff"   # purple
C_EMA        = "#ffa657"   # orange
C_THRESHOLD  = "#f85149"   # red
C_ACCEPTED   = "#3fb950"   # green
C_REJECTED   = "#f85149"   # red
C_PREPEAK    = "#ffa657"   # orange
C_NEUTRAL    = "#8b949e"   # grey

SR = 44100

# ── Helper: synthetic signal ──────────────────────────────────────────────────
def make_signal(duration=1.0, bounce_times=(0.15, 0.45, 0.75), noise_level=0.03, sr=SR):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    sig = np.random.normal(0, noise_level, len(t))
    for bt in bounce_times:
        idx = int(bt * sr)
        decay = 0.003 * sr          # exponential tail length in samples
        width = 0.0003 * sr         # sharp attack width in samples
        peak_amp = 0.9
        for k, i in enumerate(range(idx, min(idx + int(0.04 * sr), len(t)))):
            att = np.exp(-k / width) if k < width else 0
            dec = np.exp(-k / decay)
            sig[i] += peak_amp * (att + dec) * np.random.choice([-1, 1])
    return t, sig

def highpass(sig, cutoff=10000, sr=SR, order=5):
    b, a = butter(order, cutoff / (sr / 2), btype='high')
    return filtfilt(b, a, sig)

def frame_energy(sig, frame_ms=1.0, sr=SR):
    frame_len = round(sr * frame_ms / 1000)
    n = len(sig) // frame_len
    frames = sig[:n * frame_len].reshape(n, frame_len)
    return np.mean(np.abs(frames), axis=1)

def ema(energy, decay=0.9):
    avg = np.zeros_like(energy)
    avg[0] = energy[0]
    for i in range(1, len(energy)):
        avg[i] = decay * avg[i-1] + (1 - decay) * energy[i]
    return avg

def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")
def get_gt(fname, csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, 'full.csv')
    df = pd.read_csv(csv_path)
    return sorted(df[df['original-file'] == fname]['timestamp'].tolist())

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 – Raw vs High-Pass filtered signal
# ─────────────────────────────────────────────────────────────────────────────
def graph_raw_vs_filtered():
    waveform ,sr = open_audio(RAW_DIR / '01.wav')
    waveform_hp = highpass_filter(waveform)


    t_start, t_end = 2.75, 5  # seconds
    s_start, s_end = int(t_start * sr), int(t_end * sr)

    time = np.arange(s_start, s_end) / sr

    fig, axes = plt.subplots(1, 1, figsize=(12, 5), sharex=True)
    #fig.suptitle("Signal Preprocessing: Raw vs High-Pass Filtered (10 kHz)", fontweight='bold')


    axes.plot(time, waveform[s_start:s_end], linewidth=0.5, alpha=0.5, label='Original',color=C_RAW)
    axes.plot(time, waveform_hp[s_start:s_end], linewidth=0.5, alpha=0.8, label='Filtered (>10 kHz)',color=C_FILTERED)
    axes.set_ylabel('Amplitude')
    axes.set_xlabel('Time (s)')
    axes.axhline(0, color=C_NEUTRAL, linewidth=0.4, linestyle='--')
    axes.legend()
    axes.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "01_raw_vs_filtered.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 – Frame energy envelope
# ─────────────────────────────────────────────────────────────────────────────
def graph_energy_method_comparison():
    """Compare Simple/MAE, RMS, Power and Teager-Kaiser energy calculators
    using DecayAverageDetect evaluated on all STE raw files.
    Shows Precision, Recall and F1 as grouped bars."""
    import glob as _glob
    sys.path.insert(0, str(BASE / "src"))
    from detectors.energy_calculator import (SimpleEnergyCalculator,
                                             RMSEnergyCalculator,
                                             PowerEnergyCalculator,
                                             TeagerEnergyCalculator)
    from detectors.DADetector import DecayAverageDetect
    from detectors.base import evaluate_detector

    RAW_DIR_STE = BASE / "data" / "raw_sounds"
    CSV_FULL    = BASE / "data" / "full.csv"

    methods = {
        'MAE\n(Simple)': SimpleEnergyCalculator(),
        'RMS':           RMSEnergyCalculator(),
        'Power':         PowerEnergyCalculator(),
        'Teager-\nKaiser': TeagerEnergyCalculator(),
    }

    # Same file set as the detector comparison in the notebook
    _numeric = sorted([f for f in os.listdir(str(RAW_DIR_STE))
                       if f.startswith('0') and f.split('.')[0].isdigit()])
    _backhand = ['backhand-backspin-01.wav', 'backhand-backspin-02.wav',
                 'backhand-backspin-03.wav', 'backhand-backspin-05.wav']
    audio_files = [str(RAW_DIR_STE / f) for f in _numeric + _backhand
                   if (RAW_DIR_STE / f).exists()]

    results = {name: {'precision': [], 'recall': [], 'f1': []}
               for name in methods}

    for audio_path in audio_files:
        for name, calc in methods.items():
            det = DecayAverageDetect(energy_calculator=calc,
                                     pre_peak_ratio=18.0, pre_peak_window=8)
            try:
                m = evaluate_detector(det, audio_path, str(CSV_FULL), filter_noise=True)
                results[name]['precision'].append(m['precision'])
                results[name]['recall'].append(m['recall'])
                results[name]['f1'].append(m['f1'])
            except Exception:
                pass

    labels  = list(methods.keys())
    metrics = ['precision', 'recall', 'f1']
    colors  = [C_RAW, C_FILTERED, C_ENERGY]  # blue, green, purple
    display = ['Precision', 'Recall', 'F1']

    x      = np.arange(len(labels))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    for idx, (metric, col, disp) in enumerate(zip(metrics, colors, display)):
        vals = [np.mean(results[n][metric]) for n in labels]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=disp, color=col,
                      edgecolor='#21262d', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f'{v:.2f}', ha='center', va='bottom',
                    fontsize=8, color='#e6edf3')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    #ax.set_title("Energy Calculation Method Comparison  (DecayAverage detector, avg over 13 annotated files)")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linewidth=0.4)
    fig.tight_layout()
    savefig(fig, "02b_energy_method_comparison.png")


def graph_energy_envelope():
    from detectors.energy_calculator import SimpleEnergyCalculator
    waveform, sr = open_audio(RAW_DIR / '01.wav')

    time_full = np.arange(len(waveform)) / sr
    hp_frame_energy = SimpleEnergyCalculator().compute_frame_energy(
        highpass_filter(waveform), sr, frame_ms=1.0
    )

    frame_time_hp = np.arange(len(hp_frame_energy)) * 1.0 / 1000.0

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(frame_time_hp, hp_frame_energy, linewidth=0.5, color=C_ENERGY)
    ax.set_ylabel("Mean |Amplitude|")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "02_energy_envelope.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 3 – EMA tracking + threshold
# ─────────────────────────────────────────────────────────────────────────────
def graph_ema_threshold():
    t, raw = make_signal(duration=0.8, bounce_times=(0.15, 0.40, 0.65))
    
    filt = highpass(raw)
    energy = frame_energy(filt)
    avg = ema(energy, decay=0.9)
    threshold = 3.0 * avg
    n_frames = len(energy)
    frame_t = np.linspace(0, len(filt)/SR*1000, n_frames)

    fig, ax = plt.subplots(figsize=(12, 5))
    #fig.suptitle("Exponential Moving Average (EMA) & Adaptive Threshold", fontweight='bold')

    ax.fill_between(frame_t, energy, alpha=0.2, color=C_ENERGY, label='Frame Energy')
    ax.plot(frame_t, energy, color=C_ENERGY, linewidth=0.8)
    ax.plot(frame_t, avg, color=C_EMA, linewidth=1.5, label='EMA')
    ax.plot(frame_t, threshold, color=C_THRESHOLD, linewidth=1.5,
            linestyle='--', label='Threshold = 3')

    # Mark detections (naively above threshold)
    hits = np.where(energy >= threshold)[0]
    if len(hits):
        ax.scatter(frame_t[hits], energy[hits], color=C_ACCEPTED, zorder=5,
                   s=40, marker='^', label='Threshold crossed')

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Energy")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    savefig(fig, "03_ema_threshold.png")

def single_ema_threshold():
    from detectors.DADetector import DecayAverageDetect
    detector = DecayAverageDetect()

    waveform, sr = open_audio(RAW_DIR / '01.wav')
    detector.detect(waveform, sr)
    print("Generating zoomed_bounce_detection.png ...")
    gt_timestamps = get_gt('01.wav')

    det = DecayAverageDetect()
    peaks = det.detect(waveform, sr)

    energy = det.energy_history_
    avg    = det.avg_history_
    thresh = det.threshold_history_

    time_arr = np.arange(len(waveform)) / sr
    t_e      = np.arange(len(energy)) * 1.0 / 1000.0

    if len(peaks) == 0:
        print("  No peaks detected — skipping.")
        return

    zoom_center = peaks[0]
    zoom_range  = (zoom_center - 0.05, zoom_center + 0.05)  # 100 ms window

    fig, axes = plt.subplots(1, 1, figsize=(12, 5), sharex=True)



    # ── Bottom: zoomed energy / EMA / threshold ───────────
    mask_e = (t_e >= zoom_range[0]) & (t_e <= zoom_range[1])
    axes.plot(t_e[mask_e], energy[mask_e], linewidth=1.0, color=C_ENERGY,
                 label='Frame Energy')
    axes.plot(t_e[mask_e], avg[mask_e],    linewidth=1.5, color=C_EMA,
                 label=r'EMA')
    axes.plot(t_e[mask_e], thresh[mask_e], linewidth=1.2, color=C_THRESHOLD,
                 linestyle='--', label='Threshold = 3 × EMA')
    axes.axvline(zoom_center, color='red', linewidth=2, alpha=0.5)
    axes.set_ylabel('Energy')
    axes.set_xlabel('Time (s)')
    #axes.set_title('Energy & Threshold (Zoomed)')
    axes.legend()
    axes.grid(True, alpha=0.3)


    plt.tight_layout()
    savefig(fig, "03_ema_threshold.png")
# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 – Why we need pre-peak: false positive illustration
# ─────────────────────────────────────────────────────────────────────────────
def graph_false_positive_motivation():
    """Show a signal where EMA threshold triggers on a gradual ramp
       (false positive) vs. a sharp spike (true positive)."""
    np.random.seed(42)
    n = 400
    t = np.arange(n)
    energy = np.random.normal(0.01, 0.002, n).clip(0)

    # --- True bounce at t=80: instantaneous sharp spike
    energy[80] = 0.28
    energy[81:85] += np.array([0.08, 0.04, 0.02, 0.01])

    # --- False positive at t=200: gradual ramp-up (background noise build-up)
    ramp_start, ramp_end = 190, 210
    energy[ramp_start:ramp_end] += np.linspace(0, 0.09, ramp_end - ramp_start)
    energy[ramp_end:ramp_end+15] += np.linspace(0.09, 0.005, 15)

    avg = ema(energy, decay=0.9)
    threshold = 3.0 * avg

    # Compute pre-peak ratio for each "detection"
    false_idx = 205  # peak of ramp
    true_idx  = 80

    pw = 8
    pre_false = energy[false_idx-pw:false_idx].mean()
    pre_true  = energy[true_idx-pw:true_idx].mean()
    ratio_false = energy[false_idx] / pre_false if pre_false > 1e-10 else 0
    ratio_true  = energy[true_idx] / pre_true  if pre_true  > 1e-10 else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Motivation for Pre-Peak Check: Distinguishing True vs. False Positives",
                 fontweight='bold')

    for ax, idx, c, label, ratio, passed in [
        (axes[0], true_idx,  C_ACCEPTED, "True Positive\n(Sharp bounce spike)", ratio_true,  True),
        (axes[1], false_idx, C_REJECTED, "False Positive\n(Gradual ramp-up)",    ratio_false, False),
    ]:
        ax.fill_between(t, energy, alpha=0.2, color=C_ENERGY)
        ax.plot(t, energy, color=C_ENERGY, linewidth=0.9)
        ax.plot(t, avg, color=C_EMA, linewidth=1.4, linestyle='-', label='EMA')
        ax.plot(t, threshold, color=C_THRESHOLD, linewidth=1.4,
                linestyle='--', label='Threshold')

        # Pre-peak window
        pw = 8
        ax.axvspan(idx-pw, idx, alpha=0.15, color=C_PREPEAK, label=f'Pre-peak window ({pw} frames)')
        ax.axvline(idx, color=c, linewidth=2, linestyle='-')
        ax.scatter([idx], [energy[idx]], color=c, s=80, zorder=6, marker='^')

        ratio_lbl = f"Ratio = {ratio:.1f}x\n({'ACCEPTED ✓' if passed else 'REJECTED ✗'} pre_peak_ratio=18)"
        ax.text(idx + 5, energy[idx]*0.95, ratio_lbl,
                color=c, fontsize=9, va='top')
        ax.set_title(label, color=c)
        ax.set_xlabel("Frame index")
        ax.grid(True, linewidth=0.4)

    axes[0].set_ylabel("Energy")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "04_false_positive_motivation.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 5 – Pre-peak algorithm step-by-step
# ─────────────────────────────────────────────────────────────────────────────
def graph_prepeak_stepbystep():
    np.random.seed(7)
    pw = 8
    # Two cases side by side, zoomed in
    cases = [
        dict(label="Case A – Sharp onset (ACCEPTED)", color=C_ACCEPTED,
             base=0.01, spike=0.30, ramp=False),
        dict(label="Case B – Gradual onset (REJECTED)", color=C_REJECTED,
             base=0.01, spike=0.12, ramp=True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pre-Peak Ratio: Step-by-Step Illustration", fontweight='bold')

    for ax, case in zip(axes, cases):
        n = pw + 15
        energy = np.random.normal(case["base"], case["base"]*0.3, n).clip(0)
        if case["ramp"]:
            energy[2:pw] += np.linspace(0, case["spike"]*0.75, pw-2)
            energy[pw] = case["spike"]
        else:
            energy[pw] = case["spike"]

        pre_mean = energy[:pw].mean()
        ratio = energy[pw] / pre_mean if pre_mean > 1e-10 else 0
        accepted = ratio >= 18.0

        ax.bar(range(n), energy, color=[C_PREPEAK if i < pw else case["color"] for i in range(n)],
               alpha=0.75, edgecolor='#21262d', linewidth=0.5)

        ax.axhline(pre_mean, color=C_PREPEAK, linewidth=1.8, linestyle='--',
                   label=f'Pre-peak mean = {pre_mean:.4f}')
        ax.axvline(pw - 0.5, color=C_NEUTRAL, linewidth=1.5, linestyle=':', alpha=0.8)

        ax.set_xlabel("Frame index (relative)")
        ax.set_ylabel("Energy")
        ax.set_title(case["label"], color=case["color"])
        ax.legend(fontsize=9)

        # Annotation box
        status = "ACCEPTED ✓" if accepted else "REJECTED ✗"
        ax.text(pw + 0.5, energy[pw] * 0.5,
                f"e = {energy[pw]:.4f}\npre_mean = {pre_mean:.4f}\n"
                f"ratio = {ratio:.1f}x\n→ {status}",
                color=case["color"], fontsize=9.5,
                bbox=dict(facecolor="#161b22", edgecolor=case["color"],
                          boxstyle="round,pad=0.4", alpha=0.9))
        ax.grid(True, axis='y', linewidth=0.4)

    fig.tight_layout()
    savefig(fig, "05_prepeak_stepbystep.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 6 – Detection with vs without pre-peak (side-by-side comparison)
# ─────────────────────────────────────────────────────────────────────────────
def graph_with_without_prepeak():
    np.random.seed(13)
    n = 600
    energy = np.random.normal(0.008, 0.0015, n).clip(0)

    # True bounces (sharp)
    true_bounces = [80, 220, 420, 560]
    for b in true_bounces:
        energy[b]   = 0.28
        energy[b+1] = 0.10
        energy[b+2] = 0.04

    # False positives (gradual ramps)
    ramps = [140, 320, 490]

    for r in ramps:
        energy[r-10:r] += np.linspace(0, 0.07, 10)
        energy[r:r+5]  += np.linspace(0.07, 0.01, 5)

    avg = ema(energy, decay=0.9)
    threshold = 3.0 * avg
    pw, ratio_th = 8, 18.0

    # Detections WITHOUT pre-peak
    raw_hits, last = [], -100
    for i in range(len(energy)):
        if energy[i] >= threshold[i] and i - last >= 100:
            raw_hits.append(i)
            last = i

    # Detections WITH pre-peak
    filtered_hits, last = [], -100
    for i in range(pw, len(energy)):
        if energy[i] >= threshold[i] and i - last >= 100:
            pre_mean = energy[i-pw:i].mean()
            if pre_mean > 1e-10 and energy[i] / pre_mean >= ratio_th:
                filtered_hits.append(i)
                last = i

    t = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Effect of Pre-Peak Filter on Detection Results", fontweight='bold')

    for ax, hits, title in [
        (axes[0], raw_hits,      "Without Pre-Peak Filter"),
        (axes[1], filtered_hits, "With Pre-Peak Filter  (pre_peak_ratio=18, window=8)"),
    ]:
        ax.fill_between(t, energy, alpha=0.15, color=C_ENERGY)
        ax.plot(t, energy, color=C_ENERGY, linewidth=0.7, label='Energy')
        ax.plot(t, threshold, color=C_THRESHOLD, linewidth=1.2,
                linestyle='--', label='Threshold', alpha=0.8)

        # Colour-code hits: green if true bounce, red if false
        for h in hits:
            is_true = any(abs(h - b) <= 5 for b in true_bounces)
            c = C_ACCEPTED if is_true else C_REJECTED
            ax.axvline(h, color=c, linewidth=2, alpha=0.85)
            ax.scatter([h], [energy[h]], color=c, s=70, zorder=6, marker='^')

        tp = sum(1 for h in hits if any(abs(h-b)<=5 for b in true_bounces))
        fp = len(hits) - tp
        ax.set_title(f"{title}   |   Detections: {len(hits)}  "
                     f"(TP={tp}  FP={fp})", color='#e6edf3')
        ax.set_ylabel("Energy")
        ax.grid(True, linewidth=0.4)

        tp_patch  = mpatches.Patch(color=C_ACCEPTED, label='True Positive')
        fp_patch  = mpatches.Patch(color=C_REJECTED, label='False Positive')
        en_line   = mpatches.Patch(color=C_ENERGY,   label='Frame Energy', alpha=0.5)
        thr_line  = mpatches.Patch(color=C_THRESHOLD,label='Threshold',    alpha=0.8)
        ax.legend(handles=[en_line, thr_line, tp_patch, fp_patch],
                  fontsize=8, loc='upper right')

    axes[1].set_xlabel("Frame index")
    fig.tight_layout()
    savefig(fig, "06_with_without_prepeak.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 7 – Pre-peak ratio sensitivity (parameter sweep)
# ─────────────────────────────────────────────────────────────────────────────
def graph_ratio_sensitivity():
    """Show how changing pre_peak_ratio affects TP / FP / FN."""
    np.random.seed(13)
    n = 1200
    energy = np.random.normal(0.008, 0.0015, n).clip(0)

    true_bounces = [80, 220, 420, 560, 700, 900, 1050]
    for b in true_bounces:
        energy[b]   = 0.28
        energy[b+1] = 0.10
        energy[b+2] = 0.04

    ramps = [150, 310, 470, 640, 810, 975]
    for r in ramps:
        energy[r-10:r] += np.linspace(0, 0.07, 10)
        energy[r:r+5]  += np.linspace(0.07, 0.01, 5)

    avg = ema(energy, decay=0.9)
    threshold = 3.0 * avg
    pw = 8

    ratios = np.arange(1, 35, 1)
    TPs, FPs, FNs = [], [], []

    for rth in ratios:
        hits, last = [], -100
        for i in range(pw, len(energy)):
            if energy[i] >= threshold[i] and i - last >= 100:
                pre_mean = energy[i-pw:i].mean()
                if pre_mean <= 1e-10 or energy[i] / pre_mean >= rth:
                    hits.append(i)
                    last = i

        tp = sum(1 for h in hits if any(abs(h-b)<=5 for b in true_bounces))
        fp = len(hits) - tp
        fn = len(true_bounces) - tp
        TPs.append(tp); FPs.append(fp); FNs.append(fn)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Pre-Peak Ratio Sensitivity Analysis", fontweight='bold')
    ax.plot(ratios, TPs, color=C_ACCEPTED, linewidth=2, marker='o', markersize=3, label='True Positives')
    ax.plot(ratios, FPs, color=C_REJECTED, linewidth=2, marker='s', markersize=3, label='False Positives')
    ax.plot(ratios, FNs, color=C_EMA,      linewidth=2, marker='^', markersize=3, label='False Negatives (missed)')
    ax.axvline(18, color='#e6edf3', linewidth=1.5, linestyle='--', alpha=0.7, label='Default value = 18')
    ax.set_xlabel("pre_peak_ratio")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    ax.set_xlim(1, 34)
    fig.tight_layout()
    savefig(fig, "07_ratio_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 8 – Window size sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def graph_window_sensitivity():
    np.random.seed(13)
    n = 1200
    energy = np.random.normal(0.008, 0.0015, n).clip(0)

    true_bounces = [80, 220, 420, 560, 700, 900, 1050]
    for b in true_bounces:
        energy[b]   = 0.28
        energy[b+1] = 0.10
        energy[b+2] = 0.04

    ramps = [150, 310, 470, 640, 810, 975]
    for r in ramps:
        energy[r-10:r] += np.linspace(0, 0.07, 10)
        energy[r:r+5]  += np.linspace(0.07, 0.01, 5)

    avg = ema(energy, decay=0.9)
    threshold = 3.0 * avg

    windows = range(1, 25)
    TPs, FPs, FNs = [], [], []
    rth = 18.0

    for pw in windows:
        hits, last = [], -100
        for i in range(pw, len(energy)):
            if energy[i] >= threshold[i] and i - last >= 100:
                pre_mean = energy[i-pw:i].mean()
                if pre_mean <= 1e-10 or energy[i] / pre_mean >= rth:
                    hits.append(i)
                    last = i

        tp = sum(1 for h in hits if any(abs(h-b)<=5 for b in true_bounces))
        fp = len(hits) - tp
        fn = len(true_bounces) - tp
        TPs.append(tp); FPs.append(fp); FNs.append(fn)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Pre-Peak Window Size Sensitivity Analysis", fontweight='bold')
    ax.plot(list(windows), TPs, color=C_ACCEPTED, linewidth=2, marker='o', markersize=3, label='True Positives')
    ax.plot(list(windows), FPs, color=C_REJECTED, linewidth=2, marker='s', markersize=3, label='False Positives')
    ax.plot(list(windows), FNs, color=C_EMA,      linewidth=2, marker='^', markersize=3, label='False Negatives (missed)')
    ax.axvline(8, color='#e6edf3', linewidth=1.5, linestyle='--', alpha=0.7, label='Default value = 8 ms')
    ax.set_xlabel("pre_peak_window (ms frames)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    ax.set_xlim(1, 24)
    fig.tight_layout()
    savefig(fig, "08_window_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 9 – Full pipeline overview (annotated energy trace)
# ─────────────────────────────────────────────────────────────────────────────
def graph_full_pipeline():
    np.random.seed(21)
    n = 700
    energy = np.random.normal(0.008, 0.0015, n).clip(0)

    true_bounces = [100, 280, 460, 620]
    for b in true_bounces:
        energy[b]   = 0.29
        energy[b+1] = 0.11
        energy[b+2] = 0.04

    # One gradual ramp
    energy[180:195] += np.linspace(0, 0.065, 15)
    energy[195:200] += np.linspace(0.065, 0.005, 5)

    avg = ema(energy, decay=0.9)
    threshold = 3.0 * avg
    pw, rth = 8, 18.0

    final_hits = []
    last = -100
    for i in range(pw, len(energy)):
        if energy[i] >= threshold[i] and i - last >= 100:
            pre_mean = energy[i-pw:i].mean()
            if pre_mean <= 1e-10 or energy[i] / pre_mean >= rth:
                final_hits.append(i)
                last = i

    rejected = []
    last = -100
    for i in range(pw, len(energy)):
        if energy[i] >= threshold[i] and i - last >= 100:
            pre_mean = energy[i-pw:i].mean()
            if pre_mean > 1e-10 and energy[i] / pre_mean < rth:
                rejected.append(i)
                last = i

    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Full Detection Pipeline — Annotated Signal Trace", fontweight='bold')

    ax.fill_between(t, energy, alpha=0.18, color=C_ENERGY)
    ax.plot(t, energy,    color=C_ENERGY,    linewidth=0.8, label='Frame energy (post HP filter)')
    ax.plot(t, avg,       color=C_EMA,       linewidth=1.4, label='EMA  (γ=0.9)')
    ax.plot(t, threshold, color=C_THRESHOLD, linewidth=1.2, linestyle='--', label='Threshold (3×EMA)')

    for h in final_hits:
        ax.axvline(h, color=C_ACCEPTED, linewidth=2.5, alpha=0.8)
        ax.scatter([h], [energy[h]], color=C_ACCEPTED, s=90, zorder=7, marker='^')
    for r in rejected:
        ax.axvline(r, color=C_REJECTED, linewidth=2.5, alpha=0.8, linestyle=':')
        ax.scatter([r], [energy[r]], color=C_REJECTED, s=90, zorder=7, marker='x')

    acc_p = mpatches.Patch(color=C_ACCEPTED, label='Accepted detection (pre-peak ✓)')
    rej_p = mpatches.Patch(color=C_REJECTED, label='Rejected by pre-peak ✗')
    ax.legend(handles=[
        mpatches.Patch(color=C_ENERGY,    label='Frame energy', alpha=0.6),
        mpatches.Patch(color=C_EMA,       label='EMA'),
        mpatches.Patch(color=C_THRESHOLD, label='Threshold'),
        acc_p, rej_p,
    ], fontsize=8, loc='upper right')

    ax.set_xlabel("Frame index (1 frame = 1 ms)")
    ax.set_ylabel("Energy")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    savefig(fig, "09_full_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 10 – Pipeline block diagram (matplotlib-drawn)
# ─────────────────────────────────────────────────────────────────────────────
def graph_pipeline_diagram_no_prepeak():
    """Same as graph_pipeline_diagram but without the Pre-Peak Ratio Check block."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    blocks = [
        (1.0,  "Raw\nAudio"),
        (3.0,  "High-Pass\nFilter\n(10 kHz)"),
        (5.0,  "Frame\nEnergy\n(1 ms)"),
        (7.0,  "EMA\n(γ=0.9, 3×EMA)"),
        (9.0,  "Bounce\nTimestamp"),
    ]
    colors = [C_RAW, C_FILTERED, C_ENERGY, C_EMA, C_ACCEPTED]
    bw, bh = 1.5, 1.4

    for (x, label), col in zip(blocks, colors):
        rect = mpatches.FancyBboxPatch(
            (x - bw/2, 2 - bh/2), bw, bh,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=col, facecolor="#161b22"
        )
        ax.add_patch(rect)
        ax.text(x, 2, label, ha='center', va='center',
                fontsize=12, color=col, fontweight='bold')

    for i in range(len(blocks) - 1):
        x1 = blocks[i][0]   + bw/2 + 0.1
        x2 = blocks[i+1][0] - bw/2 - 0.1
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))

    fig.tight_layout()
    savefig(fig, "00b_pipeline_diagram_no_prepeak.png")


def graph_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    #fig.suptitle("DecayAverageDetect – Signal Processing Pipeline", fontweight='bold', y=0.98)

    blocks = [
        (1.0,  "Raw\nAudio"),
        (3.0,  "High-Pass\nFilter\n(10 kHz)"),
        (5.0,  "Frame\nEnergy\n(1 ms)"),
        (7.0,  "EMA\n(γ=0.9,3xEMA)"),
        (9.0,  "Pre-Peak\nRatio Check"),
        (11.0, "Bounce\nTimestamp"),
    ]
    colors = [C_RAW, C_FILTERED, C_ENERGY, C_EMA, C_PREPEAK, C_ACCEPTED]
    bw, bh = 1.5, 1.4

    for (x, label), col in zip(blocks, colors):
        rect = mpatches.FancyBboxPatch(
            (x - bw/2, 2 - bh/2), bw, bh,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=col, facecolor="#161b22"
        )
        ax.add_patch(rect)
        ax.text(x, 2, label, ha='center', va='center',
                fontsize=12, color=col, fontweight='bold')

    # Arrows
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0]   + bw/2 + 0.1
        x2 = blocks[i+1][0] - bw/2 - 0.1
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))

    # Rejection arrow from pre-peak
    pre_x = blocks[4][0]
    ax.annotate('', xy=(pre_x, 0.5), xytext=(pre_x, 2 - bh/2),
                arrowprops=dict(arrowstyle='->', color=C_REJECTED, lw=1.5))
    ax.text(pre_x + 0.15, 0.35, 'Reject\n(ratio < threshold)', color=C_REJECTED,
            fontsize=12, va='top')

    fig.tight_layout()
    savefig(fig, "00_pipeline_diagram.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 10 – Our Decay Average vs Paper Decay Average vs SuperFlux
# Uses evaluate_detector() from detectors/base.py (filter_noise=True) so that
# predictions landing on noise-labelled timestamps stay as FP and are never
# matched against valid GT — same logic as the rest of the project.
# ─────────────────────────────────────────────────────────────────────────────
def _run_comparison(raw_files, csv_path, tolerances):
    """
    Core comparison helper shared by both graph variants.

    Uses evaluate_detector(filter_noise=True) from detectors/base.py:
      - noise-labelled GT rows are excluded from the valid GT set
      - predictions near a noise timestamp remain FP (never become TP)
      - audio is loaded via audio_utils.open_audio (handles resampling)

    The paper TtDetector is file-based, so it gets a thin wrapper that
    delegates detect() back to its own file path.

    Returns dict: {tol: {det_key: [{'precision', 'recall', 'f1'}, ...]}}
    """
    import os

    for p in [str(BASE / "tt_sounds" / "src"), str(BASE / "src"), str(BASE)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from detector import Detector as TtDetector
    from detectors import SuperFluxDetect, DecayAverageDetect
    from detectors.base import BounceDetector, evaluate_detector

    class _PaperDetectorWrapper(BounceDetector):
        """Wraps the file-based TtDetector so evaluate_detector can call it."""
        def __init__(self, fpath):
            self._tt = TtDetector()
            self._fpath = fpath

        def detect(self, waveform, sr=44100):
            frames, probs, _, _, _ = self._tt.DecayAverageDetect(self._fpath)
            return [f / 1000.0 for f, p in zip(frames, probs) if p == 1]

    our_det = DecayAverageDetect()
    sf_det  = SuperFluxDetect(n_fft=1024, n_bands=24, delta=1.0, threshold_multiplier=1.0)

    det_keys = ['Decay Average (paper)', 'Decay Average (pre-peak)', 'SuperFlux']
    results  = {tol: {k: [] for k in det_keys} for tol in tolerances}

    for fname in raw_files:
        fpath = str(RAW_DIR / fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fname} not found")
            continue

        paper_wrapper = _PaperDetectorWrapper(fpath)

        for tol in tolerances:
            for key, det in [
                ('Decay Average (paper)',   paper_wrapper),
                ('Decay Average (pre-peak)', our_det),
                ('SuperFlux',               sf_det),
            ]:
                try:
                    m = evaluate_detector(det, fpath, str(csv_path),
                                          tolerance_ms=tol, filter_noise=True)
                    results[tol][key].append(m)
                except Exception as e:
                    print(f"  [warn] {fname} / {key} @ {tol}ms — {e}")

    return results


def _plot_comparison(results, tolerances, n_files, out_name):
    det_keys = ['Decay Average (paper)', 'Decay Average (pre-peak)', 'SuperFlux']
    colors   = [C_RAW, C_ACCEPTED, C_EMA]   # blue, green, orange
    metrics  = ['precision', 'recall', 'f1']
    labels   = ['Precision', 'Recall', 'F1']

    fig, axes = plt.subplots(1, len(tolerances), figsize=(7 * len(tolerances), 6))
    if len(tolerances) == 1:
        axes = [axes]

    x      = np.arange(len(metrics))
    bar_w  = 0.22
    n      = len(det_keys)
    offs   = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bar_w

    for ax, tol in zip(axes, tolerances):
        for name, col, off in zip(det_keys, colors, offs):
            rows = results[tol][name]
            if not rows:
                continue
            vals = [np.mean([r[m] for r in rows]) for m in metrics]
            bars = ax.bar(x + off, vals, bar_w, label=name,
                          color=col, alpha=0.82, edgecolor='#21262d', linewidth=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=8.5, color=col)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"Tolerance = {tol:.0f} ms  (n={n_files})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', linewidth=0.4)
        ax.axhline(1.0, color=C_NEUTRAL, linewidth=0.6, linestyle='--', alpha=0.5)

    fig.tight_layout()
    savefig(fig, out_name)


def graph_superflux_vs_decay_comparison():
    """Numbered files + backhand-backspin subset."""
    import os
    raw_files = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.startswith('0') and f.split('.')[0].isdigit()
    ]) + ['backhand-backspin-01.wav', 'backhand-backspin-02.wav',
          'backhand-backspin-03.wav', 'backhand-backspin-05.wav']

    tolerances = [5.0, 20.0]
    results = _run_comparison(raw_files, CSV_PATH, tolerances)
    _plot_comparison(results, tolerances, len(raw_files), "10_superflux_vs_decay_comparison.png")


def graph_superflux_vs_decay_comparison_b():
    """All annotated files that exist on disk."""
    df_gt = pd.read_csv(CSV_PATH)
    raw_files = [f for f in sorted(df_gt['original-file'].unique())
                 if (RAW_DIR / f).exists()]

    tolerances = [5.0, 20.0]
    results = _run_comparison(raw_files, CSV_PATH, tolerances)
    _plot_comparison(results, tolerances, len(raw_files), "10b_superflux_vs_decay_comparison_all.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 11 – Full end-to-end pipeline block diagram
# ─────────────────────────────────────────────────────────────────────────────
def graph_full_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4)
    ax.axis('off')
    #fig.suptitle("Full Detection Pipeline", fontweight='bold', y=0.98)

    C_GREEN = "#4ade80"
    C_TEAL  = "#2dd4bf"

    blocks = [
        (1.5,  "Raw\nAudio"),
        (3.8,  "Peak\nDetection"),
        (6.1,  "Extracting 15 ms\n Frames"),
        (8.4,  "Mel\nSpectrogram"),
        (10.7, "CNN"),
    ]
    colors = [C_RAW, C_ENERGY, C_GREEN, C_TEAL, C_THRESHOLD]
    bw, bh = 1.7, 1.4

    for (x, label), col in zip(blocks, colors):
        rect = mpatches.FancyBboxPatch(
            (x - bw/2, 2 - bh/2), bw, bh,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=col, facecolor="#161b22"
        )
        ax.add_patch(rect)
        ax.text(x, 2, label, ha='center', va='center',
                fontsize=12, color=col, fontweight='bold')

    for i in range(len(blocks) - 1):
        x1 = blocks[i][0]   + bw/2 + 0.1
        x2 = blocks[i+1][0] - bw/2 - 0.1
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))

    # Sub-labels below each block
#    sublabels = [
#        "44.1 kHz PCM", "Signal pipeline", "Around peak", "Fixed window",
#        "128 mel bins", "Bounce / Other"
#    ]
#    for (x, _), sub in zip(blocks, sublabels):
#        ax.text(x, 2 - bh/2 - 0.22, sub, ha='center', va='top',
#                fontsize=7.5, color='#8b949e', style='italic')

    fig.tight_layout()
    savefig(fig, "11_full_pipeline_diagram.png")


def graph_prepeak_ratio_explainer():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    #fig.suptitle("Pre-Peak Ratio: True Bounce vs False Positive", fontweight='bold')

    # --- LEFT: True bounce ---
    ax = axes[0]
    ax.set_title("True bounce — Sharp spike from silence", color=C_ACCEPTED)

    pre_energy = [0.001, 0.02, 0.04, 0.03, 0.01, 0.02, 0.01, 0.02]
    spike = 0.50
    all_energy = pre_energy + [spike]
    x = np.arange(len(all_energy))

    colors = [C_PREPEAK] * 8 + [C_ACCEPTED]
    ax.bar(x, all_energy, color=colors, width=0.7, alpha=0.75,
           edgecolor='#21262d', linewidth=0.5)

    pre_mean = np.mean(pre_energy)
    ratio = spike / pre_mean

    ax.axhline(y=pre_mean, color=C_PREPEAK, linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Pre-peak mean = {pre_mean:.3f}')
    ax.axvline(7.5, color=C_NEUTRAL, linewidth=1.5, linestyle=':', alpha=0.6)

    ax.annotate(f'E = {spike}', xy=(8, spike), xytext=(6.5, spike + 0.05),
                fontsize=10, fontweight='bold', color=C_ACCEPTED,
                arrowprops=dict(arrowstyle='->', color=C_ACCEPTED, lw=1.2))

    ax.text(4, 0.38,
            f'R = {ratio:.0f} > 18\nACCEPTED',
            fontsize=11, fontweight='bold', color=C_ACCEPTED, ha='center',
            bbox=dict(facecolor='#161b22', edgecolor=C_ACCEPTED,
                      boxstyle='round,pad=0.5', alpha=0.9))

    ax.annotate('', xy=(0, -0.04), xytext=(7, -0.04),
                arrowprops=dict(arrowstyle='<->', color=C_NEUTRAL, lw=1))
    ax.text(3.5, -0.06, 'W = 8 frames', ha='center', fontsize=9, color=C_NEUTRAL)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.09, 0.60)
    ax.set_ylabel('Frame energy $E_k$')
    ax.set_xlabel('Frames')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k−{8-i}' if i < 8 else 'k' for i in range(9)], fontsize=8)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', linewidth=0.4)

    # --- RIGHT: False positive ---
    ax = axes[1]
    ax.set_title("False positive — Spike during sustained energy", color=C_REJECTED)

    pre_energy_fp = [0.14, 0.18, 0.12, 0.19, 0.15, 0.17, 0.11, 0.16]
    spike_fp = 0.5
    all_energy_fp = pre_energy_fp + [spike_fp]
    x = np.arange(len(all_energy_fp))

    colors_fp = [C_PREPEAK] * 8 + [C_REJECTED]
    ax.bar(x, all_energy_fp, color=colors_fp, width=0.7, alpha=0.75,
           edgecolor='#21262d', linewidth=0.5)

    pre_mean_fp = np.mean(pre_energy_fp)
    ratio_fp = spike_fp / pre_mean_fp

    ax.axhline(y=pre_mean_fp, color=C_PREPEAK, linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Pre-peak mean = {pre_mean_fp:.2f}')
    ax.axvline(7.5, color=C_NEUTRAL, linewidth=1.5, linestyle=':', alpha=0.6)

    ax.annotate(f'E = {spike_fp}', xy=(8, spike_fp), xytext=(6.5, spike_fp + 0.05),
                fontsize=10, fontweight='bold', color=C_REJECTED,
                arrowprops=dict(arrowstyle='->', color=C_REJECTED, lw=1.2))

    ax.text(4, 0.38,
            f'R = {ratio_fp:.1f} > 18\nREJECTED',
            fontsize=11, fontweight='bold', color=C_REJECTED, ha='center',
            bbox=dict(facecolor='#161b22', edgecolor=C_REJECTED,
                      boxstyle='round,pad=0.5', alpha=0.9))

    ax.annotate('', xy=(0, -0.025), xytext=(7, -0.025),
                arrowprops=dict(arrowstyle='<->', color=C_NEUTRAL, lw=1))
    ax.text(3.5, -0.045, 'W = 8 frames', ha='center', fontsize=9, color=C_NEUTRAL)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.07, 0.60)
    ax.set_ylabel('Frame energy $E_k$')
    ax.set_xlabel('Frames')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k−{8-i}' if i < 8 else 'k' for i in range(9)], fontsize=8)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', linewidth=0.4)

    fig.tight_layout()
    savefig(fig, "12_prepeak_ratio_explainer.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 13 – FFT frequency-spectrum analysis (by surface & by spin)
# ─────────────────────────────────────────────────────────────────────────────
def graph_fft_analysis():
    import torchaudio
    import torch

    SOUNDS_DIR = BASE / "data" / "sounds"
    df = pd.read_csv(CSV_PATH)

    SR_CLIP = 44100
    N_SAMPLES = 661

    def load_clip(bid):
        path = SOUNDS_DIR / f"{int(bid)}.wav"
        if not path.exists():
            return None
        wf, sr = torchaudio.load(str(path))
        wf = wf.mean(0).numpy()
        if len(wf) < N_SAMPLES:
            wf = np.pad(wf, (0, N_SAMPLES - len(wf)))
        else:
            wf = wf[:N_SAMPLES]
        return wf

    freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SR_CLIP)

    # ── By surface ────────────────────────────────────────────────────────────
    surface_groups = {
        'table':  df[df['surface'] == 'table'],
        'floor':  df[df['surface'] == 'floor'],
        'racket': df[df['surface'] == 'racket'],
        'other':  df[df['surface'] == 'other'],
    }
    C_BLUE = "#58a6ff"
    surf_colors = {'table': C_BLUE, 'floor': C_EMA, 'racket': C_FILTERED, 'other': C_NEUTRAL}

    # ── By spin ───────────────────────────────────────────────────────────────
    spin_groups = {
        'backspin':  df[df['spin-direction'] == 'back'],
        'no-spin':   df[df['spin-direction'] == 'none'],
        'topspin':   df[df['spin-direction'] == 'top'],
    }
    spin_colors = {'backspin': "#58a6ff", 'no-spin': C_FILTERED, 'topspin': C_REJECTED}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, groups, colors, title in [
        (axes[0], surface_groups, surf_colors, "Average Frequency Spectrum — by Surface"),
        (axes[1], spin_groups,    spin_colors, "Average Frequency Spectrum — by Spin"),
    ]:
        for label, subset in groups.items():
            clips = [load_clip(r['bounce-id']) for _, r in subset.sample(min(80, len(subset)), random_state=42).iterrows()]
            clips = [c for c in clips if c is not None]
            if not clips:
                continue
            spectra = np.array([np.abs(np.fft.rfft(c)) for c in clips])
            avg_spec = spectra.mean(axis=0)
            db_spec = 20 * np.log10(avg_spec + 1e-12)
            db_spec -= db_spec.max()   # normalise to 0 dB peak
            ax.plot(freqs / 1000, db_spec, label=label, color=colors[label], linewidth=1.8)

        ax.set_xlim(0, SR_CLIP / 2 / 1000)
        ax.set_ylim(-60, 5)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Magnitude (dB, normalised)")
        ax.set_title(title)
        ax.axvline(10, color=C_THRESHOLD, linewidth=1, linestyle='--', alpha=0.6, label='10 kHz cutoff')
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    savefig(fig, "13_fft_analysis.png")

def fft_analysis_mean():
    import audio_utils
    fig, ax = plt.subplots(figsize=(12, 5))
    spin_colors = {'back': "#58a6ff", 'none': C_FILTERED, 'top': C_REJECTED}
    spin_types = ['back', 'none', 'top']
    df = pd.read_csv(CSV_PATH)
    for spin_type in spin_types:
        df_spin = df[df['spin-direction'] == spin_type]
        if len(df_spin) == 0:
            continue
        df_spin = df_spin.sample(min(100, len(df_spin)), random_state=42)

        spectra = []
        for _, row in df_spin.iterrows():
            bid = int(row['bounce-id'])
            path = os.path.join(SOUNDS_DIR, f'{bid}.wav')
            if not os.path.exists(path):
                continue
            aud = audio_utils.open_audio(path)
            aud = audio_utils.pad_trunc(aud, 661)
            w = aud[0].squeeze().numpy()
            fft_mag = np.abs(np.fft.rfft(w)) / len(w)
            spectra.append(fft_mag)

        spectra = np.array(spectra)
        freqs = np.fft.rfftfreq(661, d=1.0 / SR)
        mean_spec = np.mean(spectra, axis=0)
        std_spec = np.std(spectra, axis=0)

        ax.plot(freqs, mean_spec, linewidth=1.5, color=spin_colors[spin_type],
                label=f'{spin_type} (n={len(spectra)})')
        ax.fill_between(freqs, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.2, color=spin_colors[spin_type])

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    #ax.set_title('Average Frequency Spectrum by Spin Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "13_fft_analysis.png")

def fft_analysis_mean_surface():
    import audio_utils
    fig, ax = plt.subplots(figsize=(12, 5))
    surf_colors = {'table': C_RAW, 'floor': C_EMA, 'racket': C_FILTERED, 'other': C_NEUTRAL}
    surf_types = ['table', 'floor', 'racket', 'other']
    df = pd.read_csv(CSV_PATH)
    for surf_type in surf_types:
        df_surf = df[df['surface'] == surf_type]
        if len(df_surf) == 0:
            continue
        df_surf = df_surf.sample(min(100, len(df_surf)), random_state=42)

        spectra = []
        for _, row in df_surf.iterrows():
            bid = int(row['bounce-id'])
            path = os.path.join(SOUNDS_DIR, f'{bid}.wav')
            if not os.path.exists(path):
                continue
            aud = audio_utils.open_audio(path)
            aud = audio_utils.pad_trunc(aud, 661)
            w = aud[0].squeeze().numpy()
            fft_mag = np.abs(np.fft.rfft(w)) / len(w)
            spectra.append(fft_mag)

        spectra = np.array(spectra)
        freqs = np.fft.rfftfreq(661, d=1.0 / SR)
        mean_spec = np.mean(spectra, axis=0)
        std_spec = np.std(spectra, axis=0)

        ax.plot(freqs, mean_spec, linewidth=1.5, color=surf_colors[surf_type],
                label=f'{surf_type} (n={len(spectra)})')
        ax.fill_between(freqs, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.2, color=surf_colors[surf_type])

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    #ax.set_title('Average Frequency Spectrum by Surface Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "13b_fft_analysis_surface.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 14 – Mel spectrograms (one per spin type)
# ─────────────────────────────────────────────────────────────────────────────
def graph_mel_spectrogram():
    import torchaudio
    import torchaudio.transforms as T
    import torch

    SOUNDS_DIR = BASE / "data" / "sounds"
    df = pd.read_csv(CSV_PATH)

    SR_CLIP   = 44100
    N_SAMPLES = 661
    N_MELS    = 64
    N_FFT     = 1024
    HOP       = 64
    WIN       = 128

    mel_transform = T.MelSpectrogram(
        sample_rate=SR_CLIP, n_fft=N_FFT, win_length=WIN,
        hop_length=HOP, n_mels=N_MELS,
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    spin_labels = ['back', 'none', 'top']
    spin_display = ['Backspin', 'No Spin', 'Topspin']
    spin_colors  = [C_RAW, C_FILTERED, C_REJECTED]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, spin, display, col in zip(axes, spin_labels, spin_display, spin_colors):
        subset = df[df['spin-direction'] == spin]
        if subset.empty:
            ax.set_visible(False)
            continue

        # pick one representative clip
        row = subset.sample(1, random_state=7).iloc[0]
        path = SOUNDS_DIR / f"{int(row['bounce-id'])}.wav"
        if not path.exists():
            ax.set_visible(False)
            continue

        wf, sr = torchaudio.load(str(path))
        wf = wf.mean(0)
        if len(wf) < N_SAMPLES:
            wf = torch.nn.functional.pad(wf, (0, N_SAMPLES - len(wf)))
        else:
            wf = wf[:N_SAMPLES]

        mel = mel_transform(wf.unsqueeze(0)).squeeze(0)
        mel_db = db_transform(mel).numpy()

        im = ax.imshow(
            mel_db, aspect='auto', origin='lower',
            extent=[0, N_SAMPLES / SR_CLIP * 1000, 0, N_MELS],
            cmap='magma',
        )
        ax.set_title(display, color=col, fontsize=13, fontweight='bold')
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Mel bin")
        ax.tick_params(labelsize=8)

    #fig.suptitle("Mel Spectrograms — 15 ms Bounce Clips", fontsize=14, color='#e6edf3')
    fig.tight_layout()
    savefig(fig, "14_mel_spectrogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 16 – Full pipeline evaluation (detector + CNN) vs test-set baseline
# ─────────────────────────────────────────────────────────────────────────────
def graph_pipeline_eval():
    """Compare classification accuracy across three conditions:
    test-set baseline, original paper detector + CNN, and new detector + CNN."""
    import torch

    _saved_path = sys.path[:]
    for p in [str(BASE / "tt_sounds" / "src")]:
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(BASE / "src"))

    from detectors.DADetector import DecayAverageDetect
    from dataset import _surface_label, _spin_label
    from classifier.cnn import AudioClassifier
    import audio_utils as _au

    sys.path[:] = _saved_path

    # Load original paper detector
    sys.path.insert(0, str(BASE / "tt_sounds" / "src"))
    from detector import Detector as _OrigDetector
    sys.path.pop(0)

    RAW_DIR_P = BASE / "data" / "raw_sounds"
    CSV_FULL  = BASE / "data" / "full.csv"
    MODEL_DIR = BASE / "models"
    CLIP_LEN  = 661

    df_full = pd.read_csv(str(CSV_FULL))
    _ann = df_full.groupby('original-file').filter(lambda x: len(x) >= 5)['original-file'].unique()
    raw_files = [str(RAW_DIR_P / f) for f in sorted(_ann) if (RAW_DIR_P / f).exists()]

    surf_model = AudioClassifier.load_from_checkpoint(str(MODEL_DIR / "surface_best.ckpt"))
    spin_model = AudioClassifier.load_from_checkpoint(str(MODEL_DIR / "spin_best.ckpt"), strict=False)
    surf_model.eval(); spin_model.eval()

    new_det  = DecayAverageDetect(pre_peak_ratio=32, pre_peak_window=8)
    _orig_obj = _OrigDetector()

    def _classify_clips(detected_ts, wf_raw, sr, df_file):
        ground_truth = sorted(df_file['timestamp'].tolist())
        tol = 20.0 / 1000.0
        matched_gt = set()
        s_corr = s_tot = sp_corr = sp_tot = 0
        for pred_t in detected_ts:
            best_gi, best_d = -1, float('inf')
            for gi, gt in enumerate(ground_truth):
                d = abs(pred_t - gt)
                if d < best_d and gi not in matched_gt and d <= tol:
                    best_d, best_gi = d, gi
            if best_gi < 0:
                continue
            matched_gt.add(best_gi)
            start, end = int(pred_t * sr), int(pred_t * sr) + CLIP_LEN
            if end > len(wf_raw):
                continue
            clip = torch.tensor(wf_raw[start:end], dtype=torch.float32)
            mel  = _au.mel_spectro_gram(clip, sr)
            gt_row = df_file.iloc[best_gi]
            with torch.no_grad():
                sp = surf_model(mel.unsqueeze(0)).argmax(1).item()
                pp = spin_model(mel.unsqueeze(0)).argmax(1).item()
            s_corr  += int(sp == _surface_label(gt_row)); s_tot  += 1
            sp_corr += int(pp == _spin_label(gt_row));    sp_tot += 1
        return s_corr, s_tot, sp_corr, sp_tot

    new_sc = new_st = new_pc = new_pt = 0
    ori_sc = ori_st = ori_pc = ori_pt = 0

    for audio_path in raw_files:
        fname   = os.path.basename(audio_path)
        df_file = df_full[df_full['original-file'] == fname]
        if df_file.empty:
            continue
        wf_tensor, sr = _au.open_audio(audio_path)
        wf_raw = wf_tensor.numpy()

        # New detector
        ts_new = new_det.detect(wf_raw, sr)
        a, b, c, d = _classify_clips(ts_new, wf_raw, sr, df_file)
        new_sc += a; new_st += b; new_pc += c; new_pt += d

        # Original detector (takes file path)
        frames, probs, *_ = _orig_obj.DecayAverageDetect(audio_path)
        ts_orig = [f / 1000.0 for f, p in zip(frames, probs) if p == 1]
        a, b, c, d = _classify_clips(ts_orig, wf_raw, sr, df_file)
        ori_sc += a; ori_st += b; ori_pc += c; ori_pt += d

    surf_new  = new_sc / new_st if new_st else 0.0
    spin_new  = new_pc / new_pt if new_pt else 0.0
    surf_orig = ori_sc / ori_st if ori_st else 0.0
    spin_orig = ori_pc / ori_pt if ori_pt else 0.0

    # Test-set baseline
    SURF_BASE = 0.940
    SPIN_BASE = 0.949

    # ── Single grouped bar chart ──────────────────────────────────────────────
    categories = ['Surface Accuracy', 'Spin Accuracy']
    groups = {
        'Test set\n(pre-extracted clips)': ([SURF_BASE,  SPIN_BASE],  C_FILTERED),
        'Original paper\ndetector + CNN':  ([surf_orig, spin_orig],   C_EMA),
        'New detector\n+ CNN':             ([surf_new,  spin_new],    C_RAW),
    }

    x     = np.arange(len(categories))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (label, (vals, col)) in enumerate(groups.items()):
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=col, edgecolor='#21262d', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.1%}', ha='center', va='bottom', fontsize=9, color='#e6edf3')

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Full Pipeline vs Baseline  (n={len(raw_files)} annotated files)")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linewidth=0.4)
    fig.tight_layout()
    savefig(fig, "16_pipeline_eval.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 15 – CNN classification results
# ─────────────────────────────────────────────────────────────────────────────
def graph_cnn_results():
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

    # Temporarily ensure only src/ and BASE are in path (avoid tt_sounds clash)
    _saved_path = sys.path[:]
    for p in [str(BASE / "tt_sounds" / "src")]:
        if p in sys.path:
            sys.path.remove(p)

    from dataset import SoundDS, SURFACE_CLASSES, SPIN_CLASSES
    from classifier.cnn import AudioClassifier

    sys.path[:] = _saved_path

    MODEL_DIR  = BASE / "models"
    SOUNDS_DIR = BASE / "data" / "sounds"

    test_ds = SoundDS(str(BASE / "data" / "test.csv"), str(SOUNDS_DIR), augment=False)
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    surface_model = AudioClassifier.load_from_checkpoint(str(MODEL_DIR / "surface_best.ckpt"))
    spin_model    = AudioClassifier.load_from_checkpoint(str(MODEL_DIR / "spin_best.ckpt"), strict=False)

    def evaluate(model, loader, label_idx):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for mel, surface, spin in loader:
                logits = model(mel)
                preds.extend(logits.argmax(1).numpy())
                lbl = surface if label_idx == 0 else spin
                targets.extend(lbl.numpy())
        return np.array(preds), np.array(targets)

    surf_preds, surf_targets = evaluate(surface_model, loader, 0)
    spin_preds, spin_targets = evaluate(spin_model,    loader, 1)

    surf_acc = accuracy_score(surf_targets, surf_preds)
    spin_acc = accuracy_score(spin_targets, spin_preds)

    from matplotlib.colors import LinearSegmentedColormap
    # Purple cmap: dark bg → bright purple (matches the 02 energy envelope style)
    PURPLE_CMAP = LinearSegmentedColormap.from_list(
        'tt_purple', ['#161b22', '#3d2b6b', '#d2a8ff'], N=256
    )

    # ── Figure: left = surface confusion matrix, right = spin confusion matrix ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT: surface confusion matrix (13 classes)
    ax = axes[0]
    n_surf = len(SURFACE_CLASSES)
    cm_surf = confusion_matrix(surf_targets, surf_preds)
    cm_surf_norm = cm_surf.astype(float) / cm_surf.sum(axis=1, keepdims=True)

    im0 = ax.imshow(cm_surf_norm, cmap=PURPLE_CMAP, vmin=0, vmax=1, aspect='auto')
    short_names = [c.replace('racket_', 'R').replace('other', 'other') for c in SURFACE_CLASSES]
    ax.set_xticks(range(n_surf))
    ax.set_yticks(range(n_surf))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    #ax.set_title(f"Surface Classification  (acc = {surf_acc:.1%})")
    for i in range(n_surf):
        for j in range(n_surf):
            v = cm_surf_norm[i, j]
            txt_color = '#e6edf3' if v > 0.45 else '#8b949e'
            ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=6, color=txt_color)
    cb0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cb0.ax.yaxis.set_tick_params(color='#8b949e')

    # RIGHT: spin confusion matrix (3 classes)
    ax = axes[1]
    cm_spin = confusion_matrix(spin_targets, spin_preds)
    cm_spin_norm = cm_spin.astype(float) / cm_spin.sum(axis=1, keepdims=True)

    im1 = ax.imshow(cm_spin_norm, cmap=PURPLE_CMAP, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(SPIN_CLASSES, fontsize=11)
    ax.set_yticklabels(SPIN_CLASSES, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    #ax.set_title(f"Spin Classification  (acc = {spin_acc:.1%})")
    for i in range(3):
        for j in range(3):
            v = cm_spin_norm[i, j]
            txt_color = '#e6edf3' if v > 0.45 else '#8b949e'
            ax.text(j, i, f'{v:.2f}\n({cm_spin[i, j]})',
                    ha='center', va='center', fontsize=10, color=txt_color)
    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color='#8b949e')

    fig.tight_layout()
    savefig(fig, "15_cnn_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 16 – Decay Average (pre-peak) vs Decay Average — real audio, 3 panels
# ─────────────────────────────────────────────────────────────────────────────
def graph_decay_prepeak_vs_decay():
    """3-panel plot on a real file: Ground Truth / pre-peak / no pre-peak."""
    import sys, os
    import torchaudio

    for p in [str(BASE / "tt_sounds" / "src"), str(BASE / "src"), str(BASE)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from detectors import DecayAverageDetect
    from detectors.DADetector import DecayAverageDetect as DANoPrePeak

    fname   = "01.wav"
    fpath   = str(RAW_DIR / fname)
    waveform, sr = torchaudio.load(fpath)
    wf_np   = waveform.squeeze().numpy()
    duration = len(wf_np) / sr

    df_gt   = pd.read_csv(CSV_PATH)
    df_file = df_gt[df_gt['original-file'] == fname].copy()
    noise_mask = (
        (df_file['racket-type'].astype(str).str.lower() == 'none') &
        (df_file['spin-magnitude'].astype(str).str.lower() == 'none') &
        (df_file['spin-direction'].astype(str).str.lower() == 'none')
    ) | (df_file['surface'].astype(str).str.lower() == 'noise')
    gt_times = sorted(df_file.loc[~noise_mask, 'timestamp'].tolist())

    with_prepeak = DecayAverageDetect()
    no_prepeak   = DANoPrePeak(pre_peak_ratio=0.0)   # ratio=0 disables the gate

    pred_with = with_prepeak.detect(wf_np, int(sr))
    pred_no   = no_prepeak.detect(wf_np, int(sr))

    t = np.linspace(0, duration, len(wf_np))

    panels = [
        (gt_times,   C_FILTERED, f"Ground Truth ({len(gt_times)} bounces)"),
        (pred_with,  C_ACCEPTED, f"Decay Average + Pre-Peak ({len(pred_with)} detections)"),
        (pred_no,    C_RAW,      f"Decay Average — no pre-peak ({len(pred_no)} detections)"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 9), sharex=True)

    for ax, (times, color, title) in zip(axes, panels):
        ax.plot(t, wf_np, color='#8b949e', linewidth=0.4, alpha=0.7)
        for ts in times:
            ax.axvline(ts, color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(title, color=color, fontsize=11)
        ax.set_ylabel("Amplitude")
        ax.grid(True, linewidth=0.3)
        ax.set_ylim(-0.35, 0.35)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, "16_decay_prepeak_vs_decay.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating presentation graphs…")
    #graph_pipeline_diagram()
    #graph_raw_vs_filtered()
    #graph_energy_method_comparison()
    #graph_energy_envelope()
    graph_pipeline_diagram_no_prepeak()
    #graph_ema_threshold()
    #graph_false_positive_motivation()
    #graph_superflux_vs_decay_comparison_b()
    #graph_prepeak_stepbystep()
    #graph_with_without_prepeak()
    #graph_ratio_sensitivity()
    #graph_window_sensitivity()
    #graph_full_pipeline()
    #graph_superflux_vs_decay_comparison()
    #graph_full_pipeline_diagram()
    #graph_prepeak_ratio_explainer()
    #graph_fft_analysis()
    #fft_analysis_mean()
    #fft_analysis_mean_surface()
    #graph_mel_spectrogram()
    #graph_cnn_results()
    #graph_pipeline_eval()
    print("Done. All graphs saved to", OUT)
