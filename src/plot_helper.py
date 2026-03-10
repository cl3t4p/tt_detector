import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_timestamps(reference_times, detected_times, tolerance=0.03):
    """
    Match each reference timestamp to the nearest detected timestamp within tolerance.

    Returns a DataFrame with:
    - reference_time
    - detected_time
    - matched
    - shift_error_sec = detected - reference
    - abs_error_sec
    """
    reference_times = np.asarray(reference_times, dtype=float)
    detected_times = np.asarray(detected_times, dtype=float)

    used = np.zeros(len(detected_times), dtype=bool)
    rows = []

    for ref in reference_times:
        if len(detected_times) == 0:
            rows.append({
                "reference_time": ref,
                "detected_time": np.nan,
                "matched": False,
                "shift_error_sec": np.nan,
                "abs_error_sec": np.nan,
            })
            continue

        diffs = np.abs(detected_times - ref)

        # prevent reusing the same detection
        diffs[used] = np.inf

        best_idx = np.argmin(diffs)
        best_diff = diffs[best_idx]

        if best_diff <= tolerance:
            used[best_idx] = True
            det = detected_times[best_idx]
            rows.append({
                "reference_time": ref,
                "detected_time": det,
                "matched": True,
                "shift_error_sec": det - ref,
                "abs_error_sec": abs(det - ref),
            })
        else:
            rows.append({
                "reference_time": ref,
                "detected_time": np.nan,
                "matched": False,
                "shift_error_sec": np.nan,
                "abs_error_sec": np.nan,
            })

    df = pd.DataFrame(rows)

    false_positives = detected_times[~used]

    return df, false_positives


def plot_misses_and_shift_error(reference_times, comparison_df, audio_duration=None, title=None):
    """
    Plot over the original sound timeline:
    1) matched and missed reference bounces
    2) shift error for matched bounces
    """
    matched = comparison_df[comparison_df["matched"]]
    missed = comparison_df[~comparison_df["matched"]]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: misses along the file timeline
    axes[0].scatter(
        matched["reference_time"],
        np.ones(len(matched)),
        label="Matched bounce",
        marker="o"
    )
    axes[0].scatter(
        missed["reference_time"],
        np.ones(len(missed)),
        label="Missed bounce",
        marker="x",
        s=80
    )
    axes[0].set_ylabel("Reference events")
    axes[0].set_yticks([])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: shift error over time
    axes[1].axhline(0.0, linestyle="--")
    axes[1].plot(
        matched["reference_time"],
        matched["shift_error_sec"] * 1000.0,
        marker="o"
    )
    axes[1].set_xlabel("Time in original sound file (s)")
    axes[1].set_ylabel("Shift error (ms)")
    axes[1].grid(True, alpha=0.3)

    if audio_duration is not None:
        axes[1].set_xlim(0, audio_duration)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()
