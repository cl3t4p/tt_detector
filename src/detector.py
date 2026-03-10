import src.audio_utils as audio_utils
import numpy as np

class BounceDetector:
    """Base class for bounce detectors"""

    def detect(self, waveform: np.ndarray, sr:int = 44100) -> list[int]:
        """Return a list of detected bounce with the sample indices"""
        raise NotImplemented



def compute_frame_energy(waveform : np.ndarray,sr : int = 44100):
    frame_len = sr / 1000.0
    round_frame_len = round(frame_len)
    n_frames = round(len(waveform) // frame_len)
    waveform = waveform[:n_frames*round_frame_len]
    frames = waveform.reshape(n_frames,round_frame_len)
    energy = np.mean(np.abs(frames), axis=1)
    return energy



class DecayAverageDetect(BounceDetector):
    """
    Detect bounces using an exponential moving average (EMA)

    Energy forumla

    E_{k+1} = gamma * E_k + (1 - gamma) * e_k

    gamma : decay factor
    """

    def __init__(self,decay: float = 0.9, threshold_multiplier: float = 3.0, timeout_ms: float = 100.0,
                 apply_highpass: bool = True, highpass_cutoff: float = 10000.0):
        self.decay = decay
        self.threshold_multiplier = threshold_multiplier
        self.timeout_ms = timeout_ms
        self.apply_highpass = apply_highpass
        self.highpass_cutoff = highpass_cutoff


    def detect(self, waveform: np.ndarray, sr: int = 44100) -> list[int]:
        if self.apply_highpass:
            waveform = audio_utils.highpass_filter(waveform,sr,self.highpass_cutoff)

        energy = compute_frame_energy(waveform,sr)

        # Because the frames are 1 ms long then we can just put the timeout_ms
        timeout_frames = round(self.timeout_ms)


        # Get the first frame for the avg energy otherwise nothing
        # TODO add a pre-computation for the energy
        avg_energy = energy[0] if len(energy) > 0 else 0.0

        # List of detected peaks
        peaks = []
        last_peak = -timeout_frames

        # Visualization stuff can remove this afterwards
        self._energy_history = energy
        self._avg_history = np.zeros_like(energy)
        self._threshold_history = np.zeros_like(energy)

        for i,e in enumerate(energy):
            # Base formula of the decay
            avg_energy = self.decay * avg_energy + (1 - self.decay) * e

            # Store history
            self._avg_history[i] = avg_energy
            self._threshold_history[i] = self.threshold_multiplier * avg_energy


            # Check if the energy is greater than the activation threshold  + the timeout has expired
            if( e > self.threshold_multiplier * avg_energy and i - last_peak >= timeout_frames):
                # TODO Check if the timestamp does not have shift error
                #timestamp = (i*hop_length) / sr
                peaks.append(i)
                last_peak = i;

        return peaks

