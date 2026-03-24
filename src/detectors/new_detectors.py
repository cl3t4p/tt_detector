"""
References
----------
[1] Bello, J.P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M.,
    & Sandler, M. (2005). "A Tutorial on Onset Detection in Music Signals."
    IEEE Transactions on Speech and Audio Processing, 13(5), 1035-1047.
    https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/reading/Bello_2005.pdf

[2] Böck, S. & Widmer, G. (2013). "Maximum Filter Vibrato Suppression for
    Onset Detection." Proc. 16th Int. Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland.
    https://phenicx.upf.edu/system/files/publications/Boeck_DAFx-13.pdf
    Reference implementation: https://github.com/CPJKU/SuperFlux
"""

import numpy as np
import src.audio_utils as audio_utils

from .base import BounceDetector


# ---------------------------------------------------------------------------
# 1. Spectral Flux Detector
# ---------------------------------------------------------------------------

class SpectralFluxDetect(BounceDetector):
    """Detect bounces via half-wave-rectified spectral flux.

    Spectral flux measures the frame-to-frame increase in spectral magnitude.
    Only *positive* changes (energy onsets) are kept via half-wave
    rectification, which is the key insight from Dixon (2006) [2].

    The onset detection function (ODF) at frame n is:

        SF(n) = sum_k  H( |X(n,k)| - |X(n-1,k)| )

    where H(x) = (x + |x|) / 2 is the half-wave rectifier and X(n,k) is
    the STFT magnitude at frame n, frequency bin k.

    Peaks in the ODF are picked with an adaptive threshold equal to
    `threshold_multiplier` times a local median over `median_window` frames,
    combined with a minimum-distance timeout.

    References: [1] Section III-B, [2] Section 3.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 512,
                 threshold_multiplier: float = 1.5,
                 median_window: int = 15,
                 timeout_ms: float = 100.0,
                 apply_highpass: bool = True,
                 highpass_cutoff: float = 10000.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.threshold_multiplier = threshold_multiplier
        self.median_window = median_window
        self.timeout_ms = timeout_ms
        self.apply_highpass = apply_highpass
        self.highpass_cutoff = highpass_cutoff

    def detect(self, waveform: np.ndarray, sr: int = 44100) -> list[float]:
        if self.apply_highpass:
            waveform = audio_utils.highpass_filter(waveform, sr,
                                                   self.highpass_cutoff)

        # STFT magnitude
        # Use a Hann window as recommended in [1]
        window = np.hanning(self.n_fft)
        n_frames = 1 + (len(waveform) - self.n_fft) // self.hop_length
        mag = np.zeros((n_frames, self.n_fft // 2 + 1))
        for i in range(n_frames):
            start = i * self.hop_length
            frame = waveform[start:start + self.n_fft] * window
            spectrum = np.fft.rfft(frame)
            mag[i] = np.abs(spectrum)

        # Half-wave rectified spectral flux  [2]
        diff = np.diff(mag, axis=0)  # (n_frames-1, n_bins)
        flux = np.sum(np.maximum(diff, 0.0), axis=1)

        # Adaptive threshold via local median  [2]
        half_w = self.median_window // 2
        padded = np.pad(flux, (half_w, half_w), mode='reflect')
        threshold = np.array([
            np.median(padded[i:i + self.median_window])
            for i in range(len(flux))
        ]) * self.threshold_multiplier

        timeout_frames = int(self.timeout_ms / 1000.0 * sr / self.hop_length)

        peaks: list[float] = []
        last_peak = -timeout_frames
        for i in range(len(flux)):
            if flux[i] > threshold[i] and (i - last_peak) >= timeout_frames:
                # +1 because flux is computed from diff (offset by one frame)
                t = (i + 1) * self.hop_length / sr
                peaks.append(t)
                last_peak = i
        return peaks


# ---------------------------------------------------------------------------
# 2. SuperFlux Detector
# ---------------------------------------------------------------------------

class SuperFluxDetect(BounceDetector):
    """Detect bounces using the SuperFlux algorithm (Böck & Widmer, 2013).

    SuperFlux is an enhanced spectral flux that applies a *maximum filter*
    along the frequency axis of the spectrogram before computing the
    frame-to-frame difference. This suppresses slow spectral modulations
    (vibrato, tremolo) that cause false positives in plain spectral flux,
    reducing FP rate by up to 60 % in the original paper.

    Steps:
      1. Compute log-magnitude spectrogram with triangular filter bank
         (like Mel bands).
      2. Apply a maximum filter of width `max_filter_size` along the
         frequency axis of the *previous* frame.
      3. Compute half-wave-rectified difference between the current frame
         and the max-filtered previous frame.
      4. Pick peaks with adaptive threshold.

    Reference: [2].
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512,
                 n_bands: int = 24,
                 max_filter_size: int = 3,
                 threshold_multiplier: float = 1.4,
                 median_window: int = 15,
                 delta: float = 0.05,
                 timeout_ms: float = 100.0,
                 apply_highpass: bool = True,
                 highpass_cutoff: float = 10000.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.max_filter_size = max_filter_size
        self.threshold_multiplier = threshold_multiplier
        self.median_window = median_window
        self.delta = delta
        self.timeout_ms = timeout_ms
        self.apply_highpass = apply_highpass
        self.highpass_cutoff = highpass_cutoff

    @staticmethod
    def _mel_filterbank(sr: int, n_fft: int, n_bands: int) -> np.ndarray:
        """Build a triangular Mel filterbank matrix (n_bands x n_bins)."""
        n_bins = n_fft // 2 + 1
        low_mel = 0.0
        high_mel = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
        mel_points = np.linspace(low_mel, high_mel, n_bands + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)

        fb = np.zeros((n_bands, n_bins))
        for m in range(n_bands):
            left, center, right = bins[m], bins[m + 1], bins[m + 2]
            if center > left:
                fb[m, left:center] = np.linspace(0, 1, center - left,
                                                  endpoint=False)
            if right > center:
                fb[m, center:right + 1] = np.linspace(1, 0,
                                                       right - center + 1)
        return fb

    def detect(self, waveform: np.ndarray, sr: int = 44100) -> list[float]:
        if self.apply_highpass:
            waveform = audio_utils.highpass_filter(waveform, sr,
                                                   self.highpass_cutoff)

        window = np.hanning(self.n_fft)
        n_frames = 1 + (len(waveform) - self.n_fft) // self.hop_length
        n_bins = self.n_fft // 2 + 1

        # Compute magnitude spectrogram
        mag = np.zeros((n_frames, n_bins))
        for i in range(n_frames):
            start = i * self.hop_length
            frame = waveform[start:start + self.n_fft] * window
            mag[i] = np.abs(np.fft.rfft(frame))

        # Project onto Mel bands and take log  [4]
        fb = self._mel_filterbank(sr, self.n_fft, self.n_bands)
        mel_spec = mag @ fb.T  # (n_frames, n_bands)
        mel_spec = np.log1p(mel_spec)  # log(1 + x) for numerical safety

        # Maximum filter along frequency axis on previous frame  [4]
        from scipy.ndimage import maximum_filter1d
        half = self.max_filter_size // 2

        odf = np.zeros(n_frames - 1)
        for i in range(1, n_frames):
            prev_max = maximum_filter1d(mel_spec[i - 1],
                                        size=self.max_filter_size)
            diff = mel_spec[i] - prev_max
            odf[i - 1] = np.sum(np.maximum(diff, 0.0))

        # Adaptive median threshold with a minimum floor (delta).
        # The ODF is very sparse (median ≈ 0 for quiet signals) so without
        # a floor the threshold collapses to 0 and every non-zero frame
        # triggers a detection.
        half_w = self.median_window // 2
        padded = np.pad(odf, (half_w, half_w), mode='reflect')
        threshold = np.array([
            np.median(padded[i:i + self.median_window])
            for i in range(len(odf))
        ]) * self.threshold_multiplier + self.delta

        timeout_frames = int(self.timeout_ms / 1000.0 * sr / self.hop_length)

        # When the ODF triggers, refine the timestamp by finding the
        # highest absolute amplitude peak in the raw waveform within the
        # STFT analysis window.  This gives sample-accurate timing.
        abs_wave = np.abs(waveform)

        peaks: list[float] = []
        last_peak = -timeout_frames
        for i in range(len(odf)):
            if odf[i] > threshold[i] and (i - last_peak) >= timeout_frames:
                # Search region: the STFT frame that triggered the ODF
                search_start = (i + 1) * self.hop_length
                search_end = min(search_start + self.n_fft, len(waveform))
                peak_sample = search_start + np.argmax(
                    abs_wave[search_start:search_end]
                )
                t = peak_sample / sr
                peaks.append(t)
                last_peak = i
        return peaks