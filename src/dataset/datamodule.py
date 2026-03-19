import lightning
import os
from torch.utils.data import DataLoader
from .dataset import SoundDS



class BounceDataModule(lightning.LightningDataModule):


    def __init__(self,
                 data_dir : str = 'data',
                 sounds_subdir='sounds',
                 batch_size : int = 32,
                 num_workers : int = 4,
                 sr : int = 44100,
                 max_len : int = 611):
        super().__init__()
        self.data_dir = data_dir
        self.sounds_dir = os.path.join(data_dir,sounds_subdir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr
        self.max_len = max_len


    def setup(self,stage=None):
        train_csv = os.path.join(self.data_dir,'train.csv')
        test_csv = os.path.join(self.data_dir,'test.csv')

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
        return DataLoader(self.train_ds,batch_size=self.batch_size,
                          shuffle=True,num_workers=self.num_workers)

    
    def val_dataloader(self):
        return DataLoader(self.train_ds,batch_size=self.batch_size,
                          shuffle=False,num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
