from .base import BounceDetector,BaseEnergyCalculator
from .energy_calculator import SimpleEnergyCalculator
from collections import deque
import src.audio_utils as audio_utils
import scipy.signal as scipy_signal

import numpy as np


class AdjacentFrameDetect(BounceDetector):
    """Detect bounces by comparing adjacent frame energies.

    A bounce is flagged when the current frame's energy exceeds the
    previous frame's energy by a fixed absolute threshold. A timeout
    prevents double-triggering on the same event.
    """

    def __init__(self, threshold: float = 0.005, frame_ms: float = 1.0,
                 timeout_ms: float = 20.0,
                 energy_calc : BaseEnergyCalculator = SimpleEnergyCalculator()):
        self.threshold = threshold
        self.frame_ms = frame_ms
        self.timeout_ms = timeout_ms
        self.energy_calc = energy_calc

    def detect(self, waveform: np.ndarray, sr: int = 44100) -> list[float]:
        energy = self.energy_calc.compute_frame_energy(waveform, sr, self.frame_ms)
        timeout_frames = int(self.timeout_ms / self.frame_ms)

        peaks = []
        last_peak = -timeout_frames
        for i in range(1, len(energy)):
            if (energy[i] - energy[i - 1] > self.threshold
                    and i - last_peak >= timeout_frames):
                peaks.append(i * self.frame_ms / 1000.0)
                last_peak = i
        return peaks


class MovingAverageDetect(BounceDetector):
    """Detect bounces using a sliding window average of frame energies."""

    def __init__(self, threshold_multiplier: float = 5.0,
                 window_size: int = 100, frame_ms: float = 1.0,
                 timeout_ms: float = 50.0,
                 energy_calc : BaseEnergyCalculator = SimpleEnergyCalculator()):
        self.threshold_multiplier = threshold_multiplier
        self.window_size = window_size
        self.frame_ms = frame_ms
        self.timeout_ms = timeout_ms
        self.energy_calc = energy_calc

    def detect(self, waveform: np.ndarray, sr: int = 44100) -> list[float]:
        energy = self.energy_calc.compute_frame_energy(waveform, sr, self.frame_ms)
        timeout_frames = int(self.timeout_ms / self.frame_ms)

        window = deque(maxlen=self.window_size)
        peaks = []
        last_peak = -timeout_frames

        for i, e in enumerate(energy):
            window.append(e)
            avg = np.mean(window)
            if (e > self.threshold_multiplier * avg
                    and i - last_peak >= timeout_frames):
                peaks.append(i * self.frame_ms / 1000.0)
                last_peak = i
        return peaks


class ScipyPeakDetect(BounceDetector):
    """Detect bounces using scipy.signal.find_peaks on frame energy."""

    def __init__(self, min_distance_ms: float = 100.0, frame_ms: float = 1.0,
                 prominence: float = 0.01, apply_highpass: bool = True,
                 highpass_cutoff: float = 10000.0,
                 energy_calc: BaseEnergyCalculator = SimpleEnergyCalculator()):
        self.min_distance_ms = min_distance_ms
        self.frame_ms = frame_ms
        self.prominence = prominence
        self.apply_highpass = apply_highpass
        self.highpass_cutoff = highpass_cutoff
        self.energy_calc = energy_calc

    def detect(self, waveform: np.ndarray, sr: int = 44100) -> list[float]:
        if self.apply_highpass:
            waveform = audio_utils.highpass_filter(waveform, sr, self.highpass_cutoff)

        energy = self.energy_calc.compute_frame_energy(waveform, sr, self.frame_ms)
        min_dist = int(self.min_distance_ms / self.frame_ms)

        peak_indices, properties = scipy_signal.find_peaks(
            energy, distance=min_dist, prominence=self.prominence
        )
        return [idx * self.frame_ms / 1000.0 for idx in peak_indices]