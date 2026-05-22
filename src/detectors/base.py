import numpy as np
import pandas as pd
import audio_utils
import os


class BounceDetector:
    """Base class for bounce detectors"""

    def detect(self, waveform: np.ndarray, sr:int = 44100) -> list[float] | list[int]:
        """Return a list of detected bounce with the sample indices"""
        raise NotImplemented

class BaseEnergyCalculator:
    """Base class for energy calculation """

    def compute_frame_energy(self, waveform: np.ndarray,sr:int = 44100,frame_ms : float = 1.0) -> np.ndarray:
        raise NotImplemented
        

def _is_noise(row) -> bool:
    """Return True for entries that are noise / unlabelled table bounces.

    A row is noise when all three classification labels are 'none'
    (racket-type, spin-magnitude, spin-direction) OR when the surface
    column explicitly says 'noise'.
    """
    surface_noise = str(row.get('surface', '')).lower() == 'noise'
    unlabelled = (
        str(row.get('racket-type', 'none')).lower() == 'none'
        and str(row.get('spin-magnitude', 'none')).lower() == 'none'
        and str(row.get('spin-direction', 'none')).lower() == 'none'
    )
    return surface_noise or unlabelled


def evaluate_detector(detector: BounceDetector, raw_audio_path: str,
                      csv_path: str, sr: int = 44100,
                      tolerance_ms: float = 5.0,
                      filter_noise: bool = False):
    """Evaluate a detector against ground-truth labels.

    When *filter_noise* is True, GT entries labelled as noise
    (racket-type=spin-magnitude=spin-direction='none', or surface='noise')
    are excluded from the valid ground-truth set.  A detection that falls
    within tolerance of a noise entry is still counted as a false positive.

    Returns precision, recall, and mean onset error.
    """
    sig, orig_sr = audio_utils.open_audio(raw_audio_path)

    waveform = sig.squeeze().numpy()
    predicted = detector.detect(waveform, sr)

    raw_name = os.path.basename(raw_audio_path)
    df = pd.read_csv(csv_path)
    df_file = df[df['original-file'] == raw_name].copy()

    if filter_noise:
        noise_mask = df_file.apply(_is_noise, axis=1)
        noise_timestamps = set(df_file.loc[noise_mask, 'timestamp'].tolist())
        df_valid = df_file[~noise_mask]
    else:
        noise_timestamps = set()
        df_valid = df_file

    ground_truth = sorted(df_valid['timestamp'].tolist())

    tolerance_s = tolerance_ms / 1000.0
    matched_gt = set()
    matched_pred = set()
    onset_errors = []

    noise_list = sorted(noise_timestamps)

    for pi, p in enumerate(predicted):
        # Check whether this prediction falls within a noise timestamp.
        # If so it cannot become a TP regardless of valid-GT matching.
        if filter_noise and any(abs(p - n) <= tolerance_s for n in noise_list):
            continue  # leave as FP — do not try to match valid GT

        best_dist = float('inf')
        best_gi = -1
        for gi, gt in enumerate(ground_truth):
            d = abs(p - gt)
            if d < best_dist and gi not in matched_gt:
                best_dist = d
                best_gi = gi
        if best_dist <= tolerance_s and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            onset_errors.append(best_dist * 1000.0)  # in ms

    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mean_error = np.mean(onset_errors) if onset_errors else float('nan')

    return {
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        'mean_onset_error_ms': mean_error,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
    }
