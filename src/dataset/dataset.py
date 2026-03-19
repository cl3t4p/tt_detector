import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import audio_utils
import src.audio_utils

SURFACE_CLASSES = [
        'other',
        'racket_01',
        'racket_02',
        'racket_03',
        'racket_04',
        'racket_05',
        'racket_06',
        'racket_07',
        'racket_08',
        'racket_09',
        'racket_10',
        'table',
        'floor'
        ]

SPIN_CLASSES = [
        'back',
        'none',
        'top'
        ]


def _surface_label(row) -> int:
    """Map CSV surface/racket-type to class index"""
    surface = str(row['surface']).strip().lower()
    racket_type = str(row['racket-type']).strip().lower()

    if(surface == 'racket' and racket_type != 'none'):
        key = f'racket_{racket_type.zfill(2)}'
    else:
        key = surface

        
    if(key in SURFACE_CLASSES):
        return SURFACE_CLASSES.index(key)
    else:
        return SURFACE_CLASSES.index('other')

def _spin_label(row) -> int:
    direction = str(row['spin-direction']).strip().lower()

    if(direction in SURFACE_CLASSES):
        return SURFACE_CLASSES.index(direction)
    else:
        return SURFACE_CLASSES.index('other')



class SoundDS(Dataset):
    """Dataset of the individual bounce sound clips"""

    def __init__(self,csv_path: str,sounds_dir: str, max_len: int = 661):
        self.df = pd.read_csv(csv_path)
        self.sounds_dir = sounds_dir
        self.max_len = max_len


    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        bounce_id = int(row['bounce-id'])
        audio_path = os.path.join(self.sounds_dir,f"{bounce_id}.wav")


        # Load and preprocess
        aud = audio_utils.open_audio(audio_path)
        aud = audio_utils.pad_trunc(aud)

        waveform , sr = aud
        mel = audio_utils.mel_spectro_gram(waveform.numpy(),sr)
        surface = _surface_label(row)
        spin = _spin_label(row)

        return mel, torch.tensor(surface,dtype=torch.long), \
                torch.tensor(spin,dtype=torch.long)
