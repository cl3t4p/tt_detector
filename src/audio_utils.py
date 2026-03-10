import torchaudio
from scipy.signal import butter,sosfiltfilt
import numpy as np
import torchaudio.transforms as Transforms




def open_audio(audio_file: str):
    """Load audio file and return (mono_waveform, sample_rate)"""
    waveform, sr = torchaudio.load(audio_file)

    # Do not return dimension we work only with mono
    waveform = waveform.mean(dim=0)
    #if waveform.shape[0] != 1:
    #    waveform = waveform.mean(dim=0,keepdim=True)
    return waveform,sr


## Filters

def highpass_filter(waveform: np.ndarray,sr: int = 44100,
                    cutoff: float = 10000.0,order:int = 5) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to a waveform to isolate
    the high-frequency components of the ball bounces
    """
    butt = butter(order,cutoff,btype='highpass',fs=sr,output='sos')
    filtered = sosfiltfilt(butt,waveform)
    return filtered

## Extraction

def mel_spectro_gram(waveform:np.ndarray,sr: int = 44100,n_ftt: int = 1024
                     ,win_length: int = 128,hop_length: int = 64, 
                     n_mels: int = 64, top_db: int = 80):
    mel = Transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_ftt,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            )(waveform)
    mel_db = Transforms.AmplitudeToDB(top_db=top_db)(mel)
    return mel_db



