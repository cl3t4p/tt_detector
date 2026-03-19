import torchaudio
from scipy.signal import butter,sosfiltfilt
import numpy as np
import torch
import torchaudio.transforms as Transforms




def open_audio(audio_file: str):
    """Load audio file and return (mono_waveform, sample_rate)"""
    waveform, sr = torchaudio.load(audio_file)

    # Do not return dimension we work only with mono
    waveform = waveform.mean(dim=0)
    #if waveform.shape[0] != 1:
    #    waveform = waveform.mean(dim=0,keepdim=True)
    return waveform,sr


def pad_trunc(audio, max_len : int = 611):
    waveform , sr = audio
    num_samples  = len(waveform)
    if(num_samples > max_len):
    # Truncate from the beginning
        waveform = waveform[:max_len]
    elif num_samples < max_len:
        # Fill with zeros with a random shift
        pad_total = max_len - num_samples
        pad_begin = torch.randint(0,pad_total + 1 ,(1,)).item()
        pad_end = pad_total - pad_begin
        waveform = torch.nn.functional.pad(waveform,(pad_begin,pad_end))
    return waveform , sr



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



