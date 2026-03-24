import os

import lightning
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import src.audio_utils as audio_utils

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

    if surface == 'racket' and racket_type != 'none':
        key = f'racket_{racket_type.zfill(2)}'
    else:
        key = surface

        
    if key in SURFACE_CLASSES:
        return SURFACE_CLASSES.index(key)
    else:
        return SURFACE_CLASSES.index('other')

def _spin_label(row) -> int:
    direction = str(row['spin-direction']).strip().lower()

    if direction in SPIN_CLASSES:
        return SPIN_CLASSES.index(direction)
    else:
        return SPIN_CLASSES.index('none')



def _time_shift(audio, shift_limit: float = 0.05):
    """Randomly circular-shift a 1D waveform."""
    waveform, sr = audio

    max_shift = int(shift_limit * waveform.shape[0])
    if max_shift == 0:
        return waveform, sr

    shift_amt = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    waveform = waveform.roll(shift_amt)

    return waveform, sr


class SoundDS(Dataset):
    """Dataset of the individual bounce sound clips"""

    def __init__(self,
                 csv_path: str,
                 sounds_dir: str, 
                 max_len: int = 661,
                 augment: bool = True):
        self.df = pd.read_csv(csv_path)
        self.sounds_dir = sounds_dir
        self.max_len = max_len
        self.augment = augment


    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        bounce_id = int(row['bounce-id'])
        audio_path = os.path.join(self.sounds_dir,f"{bounce_id}.wav")


        # Load and preprocess
        aud = audio_utils.open_audio(audio_path)
        aud = audio_utils.pad_trunc(aud)

        if self.augment:
            aud = _time_shift(aud)

        waveform , sr = aud
        mel = audio_utils.mel_spectro_gram(waveform,sr)
        surface = _surface_label(row)
        spin = _spin_label(row)
            

        return mel, torch.tensor(surface,dtype=torch.long), \
                torch.tensor(spin,dtype=torch.long)


class SoundDataModule(lightning.LightningDataModule):

    def __init__(self,
                 data_dir: str = 'data',
                 sounds_subdir='sounds',
                 batch_size: int = 32,
                 num_workers: int = 4,
                 sr: int = 44100,
                 max_len: int = 611):
        super().__init__()
        self.test_ds = None
        self.train_ds = None
        self.data_dir = data_dir
        self.sounds_dir = os.path.join(data_dir, sounds_subdir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr
        self.max_len = max_len

    def setup(self, stage=None):
        train_csv = os.path.join(self.data_dir, 'train.csv')
        test_csv = os.path.join(self.data_dir, 'test.csv')

        self.train_ds = SoundDS(train_csv,
                                self.sounds_dir,
                                self.max_len,
                                augment=True
                                )
        self.test_ds = SoundDS(train_csv,
                               self.sounds_dir,
                               self.max_len,
                               augment=False
                               )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
