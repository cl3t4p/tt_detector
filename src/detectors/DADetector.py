import src.audio_utils as audio_utils
import numpy as np

from .base import BounceDetector,BaseEnergyCalculator
from .energy_calculator import SimpleEnergyCalculator





class DecayAverageDetect(BounceDetector):
    """Detect bounces using an exponential moving average (EMA) of energy.
        E_{k+1} = gamma * E_k + (1 - gamma) * e_k
    """

    def __init__(self, decay: float = 0.9, 
                 threshold_multiplier: float = 3.0,
                 frame_ms: float = 1.0, 
                 timeout_ms: float = 100.0,
                 apply_highpass: bool = True, 
                 highpass_cutoff: float = 10000.0,
                 energy_calculator : BaseEnergyCalculator = SimpleEnergyCalculator(),
                 return_indexes=True):
        self.decay = decay
        self.threshold_multiplier = threshold_multiplier
        self.frame_ms = frame_ms
        self.timeout_ms = timeout_ms
        self.apply_highpass = apply_highpass
        self.highpass_cutoff = highpass_cutoff
        self.energy_calculator = energy_calculator
        self.return_indexes = return_indexes



    def detect(self, waveform: np.ndarray, sr: int = 44100) -> list[float] | list[int]:

        # Only for graphs
        if self.apply_highpass:
            waveform = audio_utils.highpass_filter(waveform, sr, self.highpass_cutoff)

        hop_length = int(sr*self.frame_ms/1000.0)
        energy = self.energy_calculator.compute_frame_energy(waveform, sr, self.frame_ms)
        timeout_frames = int(self.timeout_ms / self.frame_ms)

        avg_energy = energy[0] if len(energy) > 0 else 0.0
        peaks = []
        last_peak = -timeout_frames
        
        # Store intermediate values for graphs
        self.energy_history_ = energy
        self.avg_history_ = np.zeros_like(energy)
        self.threshold_history_ = np.zeros_like(energy)

        for i, e in enumerate(energy):

            avg_energy = self.decay * avg_energy + (1 - self.decay) * e
            self.avg_history_[i] = avg_energy
            self.threshold_history_[i] = self.threshold_multiplier * avg_energy

            if (e >= self.threshold_multiplier * avg_energy
                    and i - last_peak >= timeout_frames):


                if(self.return_indexes):
                    peaks.append(i*hop_length)
                else:
                    timestamp = (i*hop_length) /sr
                    peaks.append(timestamp)
                
                last_peak = i
        return peaks

