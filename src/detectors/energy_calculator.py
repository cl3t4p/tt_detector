import numpy as np

from .base import BaseEnergyCalculator


class SimpleEnergyCalculator(BaseEnergyCalculator):

    def compute_frame_energy(self, waveform: np.ndarray, sr: int = 44100, frame_ms : float = 1) -> np.ndarray:
        frame_len = round(sr * frame_ms / 1000.0)
        n_frames = len(waveform) // frame_len
        waveform = waveform[:n_frames * frame_len]
        frames = waveform.reshape(n_frames, frame_len)
        energy = np.mean(np.abs(frames), axis=1)
        return energy

class RMSEnergyCalculator(BaseEnergyCalculator):
    def compute_frame_energy(
        self,
        waveform: np.ndarray,
        sr: int = 44100,
        frame_ms: float = 1.0
    ) -> np.ndarray:
        frame_len = max(1, round(sr * frame_ms / 1000.0))
        n_frames = len(waveform) // frame_len
        waveform = waveform[:n_frames * frame_len]
        frames = waveform.reshape(n_frames, frame_len)
        energy = np.sqrt(np.mean(frames ** 2, axis=1))
        return energy



class PowerEnergyCalculator(BaseEnergyCalculator):
    def compute_frame_energy(
        self,
        waveform: np.ndarray,
        sr: int = 44100,
        frame_ms: float = 1.0
    ) -> np.ndarray:
        frame_len = max(1, round(sr * frame_ms / 1000.0))
        n_frames = len(waveform) // frame_len
        waveform = waveform[:n_frames * frame_len]
        frames = waveform.reshape(n_frames, frame_len)
        energy = np.mean(frames ** 2, axis=1)
        return energy


class TeagerEnergyCalculator(BaseEnergyCalculator):
    def compute_frame_energy(
        self,
        waveform: np.ndarray,
        sr: int = 44100,
        frame_ms: float = 1.0
    ) -> np.ndarray:
        frame_len = max(1, round(sr * frame_ms / 1000.0))

        x = waveform.astype(np.float64)
        if len(x) < 3:
            return np.array([], dtype=np.float64)

        # Teager-Kaiser operator
        tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]

        n_frames = len(tkeo) // frame_len
        tkeo = tkeo[:n_frames * frame_len]
        frames = tkeo.reshape(n_frames, frame_len)

        energy = np.mean(np.abs(frames), axis=1)
        return energy
