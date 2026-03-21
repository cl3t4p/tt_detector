import numpy as np
import matplotlib.pyplot as plt


## Waveform visualization

def plot_waveform(waveform, sr=44100,title="Waveform",ax=None):
    """Plot time-domain Waveform"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    if hasattr(waveform, "numpy"):
        waveform = waveform.squeeze().numpy()
    time = np.arange(len(waveform))
    ax.plot(time, waveform,linewidth=0.5)
    ax.set_xlabel("Time (S)")
    ax.set_ylabel("Waveform (V)")
    ax.set_title(title)
    ax.grid(True,alpha=0.3)
    return ax

def plot_waveform_ms(waveform, sr=44100,title="Bounce Waveform",ax=None):
    """Plot time-domain Waveform"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,3))
    if hasattr(waveform, "numpy"):
        waveform = waveform.squeeze().numpy()
    time_ms = np.arange(len(waveform)) / sr * 1000.0
    ax.plot(time_ms, waveform,linewidth=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Waveform (V)")
    ax.set_title(title)
    ax.grid(True,alpha=0.3)
    return ax

## Spectrogram visualization

def plot_mel_spectrogram(mel_db, sr=44100, hop_length=64, title="Mel spectrogram", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    if hasattr(mel_db, "numpy"):
        mel_db = mel_db.squeeze().numpy()

    n_frames = mel_db.shape[1]
    time_ms = np.arange(n_frames)* hop_length / sr * 1000.0
    extent = [time_ms[0],time_ms[-1],0,mel_db.shape[0]]

    im = ax.imshow(mel_db,aspect="auto",origin="lower",extent=extent,cmap='inferno')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mel Frequency Bins (dB)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax,label="dB")
    return ax

def plot_spectrogram(spec,title="Spectrogram",ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    if hasattr(spec, "numpy"):
        spec = spec.squeeze().numpy()
    spec_db = 10 * np.log10(spec + 1e-10)
    im = ax.imshow(spec_db, aspect='auto', origin='lower', cmap='inferno')
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Frequency Bin")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='dB')
    return ax

def plot_mfcc(mfcc_data, sr=44100, hop_length=64, title="MFCC", ax=None):
    """Plot MFCCs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    if hasattr(mfcc_data, 'numpy'):
        mfcc_data = mfcc_data.squeeze().numpy()

    n_frames = mfcc_data.shape[1]
    time_ms = np.arange(n_frames) * hop_length / sr * 1000.0
    extent = [time_ms[0], time_ms[-1], 0, mfcc_data.shape[0]]

    im = ax.imshow(mfcc_data, aspect='auto', origin='lower', extent=extent,
                   cmap='inferno')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("MFCC Coefficient")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')
    return ax