from __future__ import annotations

import numpy as np


class SignalSpinClassifier:
    """Rule-based spin detector for a single 15 ms bounce segment.

    Usage:
        label = SignalSpinClassifier().predict(segment_15ms, sr=44100)
    """

    def __init__(
        self,
        *,
        rms_floor: float = 1e-4,
        decision_threshold: float = 0.55,
        low_peak_band: tuple[float, float] = (900.0, 2500.0),
        low_ref_band: tuple[float, float] = (3000.0, 6000.0),
        high_peak_band: tuple[float, float] = (7800.0, 12000.0),
        high_ref_band: tuple[float, float] = (5500.0, 7600.0),
        tail_band: tuple[float, float] = (12000.0, 17000.0),
        low_center: float = 1.20,
        high_center: float = 1.10,
        tail_center: float = 0.08,
        low_slope: float = 4.0,
        high_slope: float = 4.0,
        tail_slope: float = 40.0,
        low_weight: float = 0.35,
        high_weight: float = 0.45,
        tail_weight: float = 0.20,
    ) -> None:
        self.rms_floor = rms_floor
        self.decision_threshold = decision_threshold

        self.low_peak_band = low_peak_band
        self.low_ref_band = low_ref_band
        self.high_peak_band = high_peak_band
        self.high_ref_band = high_ref_band
        self.tail_band = tail_band

        self.low_center = low_center
        self.high_center = high_center
        self.tail_center = tail_center

        self.low_slope = low_slope
        self.high_slope = high_slope
        self.tail_slope = tail_slope

        self.low_weight = low_weight
        self.high_weight = high_weight
        self.tail_weight = tail_weight

    @staticmethod
    def _sigmoid(x: float, center: float, slope: float) -> float:
        return float(1.0 / (1.0 + np.exp(-slope * (x - center))))

    @staticmethod
    def _band_power(
        freqs: np.ndarray,
        spectrum_power: np.ndarray,
        band: tuple[float, float],
    ) -> float:
        low_hz, high_hz = band
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if not np.any(mask):
            return 0.0
        return float(np.sum(spectrum_power[mask]))

    def predict_with_score(
        self,
        segment_15ms: np.ndarray,
        sr: int = 44100,
    ) -> dict[str, float | str]:
        signal = np.asarray(segment_15ms, dtype=np.float64).reshape(-1)
        if signal.size == 0:
            return {"label": "no_spin", "score": 0.0}

        signal = signal - float(np.mean(signal))
        rms = float(np.sqrt(np.mean(signal**2)))
        if rms < self.rms_floor:
            return {"label": "no_spin", "score": 0.0}

        windowed = signal * np.hanning(signal.size)
        spectrum = np.fft.rfft(windowed)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / float(sr))
        total_power = float(np.sum(power)) + 1e-12

        low_peak = self._band_power(freqs, power, self.low_peak_band)
        low_ref = self._band_power(freqs, power, self.low_ref_band)
        high_peak = self._band_power(freqs, power, self.high_peak_band)
        high_ref = self._band_power(freqs, power, self.high_ref_band)
        tail = self._band_power(freqs, power, self.tail_band)

        low_contrast = low_peak / (low_ref + 1e-12)
        high_contrast = high_peak / (high_ref + 1e-12)
        tail_ratio = tail / total_power

        low_score = self._sigmoid(low_contrast, self.low_center, self.low_slope)
        high_score = self._sigmoid(high_contrast, self.high_center, self.high_slope)
        tail_score = self._sigmoid(tail_ratio, self.tail_center, self.tail_slope)

        score = (
            self.low_weight * low_score
            + self.high_weight * high_score
            + self.tail_weight * tail_score
        )
        label = "spin" if score >= self.decision_threshold else "no_spin"
        return {"label": label, "score": float(score)}

    def predict(self, segment_15ms: np.ndarray, sr: int = 44100) -> str:
        return str(self.predict_with_score(segment_15ms=segment_15ms, sr=sr)["label"])
