import numpy as np


class BounceDetector:
    """Base class for bounce detectors"""

    def detect(self, waveform: np.ndarray, sr:int = 44100) -> list[float] | list[int]:
        """Return a list of detected bounce with the sample indices"""
        raise NotImplemented

class BaseEnergyCalculator:
    """Base class for energy calculation """

    def compute_frame_energy(self, waveform: np.ndarray,sr:int = 44100,frame_ms : float = 1.0) -> np.ndarray:
        raise NotImplemented
        

