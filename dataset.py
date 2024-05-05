import os
import librosa
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset

from config import Config
from augmentation import frequency_domain_augmentations as FDA

class DatasetUtils:
    @staticmethod
    def make_melspectogram(audio_data, config: Config):
        melspec = librosa.feature.melspectrogram(
            y=audio_data, sr=config.sample_rate, n_mels=config.n_mels, 
            fmin=config.min_frequency, fmax=config.max_frequency,
        )

        return librosa.power_to_db(melspec).astype(np.float32)
    
    @staticmethod
    def convert_to_colored_image(data, eps=1e-6):
        mean = data.mean()
        std = data.std()

        data = (data - mean) / (std + eps)
        _min, _max = data.min(), data.max()

        if (_max - _min) > eps:
            image = np.clip(data, _min, _max)
            image = 255 * (image - _min) / (_max - _min)
            image = image.astype(np.uint8)
        else:
            image = np.zeros_like(data, dtype=np.uint8)
            
        return image
    
    @staticmethod
    def prepare_data(row: pd.DataFrame, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
        path = os.path.join(config.path_to_data, row['filename'])
        path = path.split('.')[0]
        path += '.npy'

        mel_spectogram = np.load(path)
        labels = torch.tensor(list(row)[2:]).float()

        return mel_spectogram, labels


class BirdCLEFDataset(Dataset):
    def __init__(self, data, config: Config, augmentations=None):
        super().__init__()
        self.data = data
        self.config = config
        self.augmentations = augmentations 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, labels = DatasetUtils.prepare_data(self.data.loc[idx], self.config)
        data = DatasetUtils.convert_to_colored_image(data)

        if self.augmentations:
            data = self.augmentations(image=data)['image']
       
            # Use mix up here
            if np.random.uniform(0, 1) < 0.3:
                random_sample_idx = np.random.randint(0, len(self.data))
                new_data, new_label = DatasetUtils.prepare_data(self.data.loc[random_sample_idx], self.config)
                new_data = self.augmentations(image=new_data)['image']
                data, labels = FDA.mix_up(data, labels, new_data, new_label)

        return torch.tensor(data, dtype=torch.float32), labels
    