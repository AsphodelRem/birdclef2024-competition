import os
import librosa
import numpy as np

import torch
from torch.utils.data import Dataset

from config import Config


class BirdCLEFDataset(Dataset):
    def __init__(self, data, path_to_data: str, config: Config):
        super().__init__()

        self.path_to_data = path_to_data
        self.data = data
        self.config = config
        
    def _make_melspec(self, audio_data):
        melspec = librosa.feature.melspectrogram(
            y=audio_data, sr=self.config.sample_rate, n_mels=self.config.n_mels, 
            fmin=self.config.min_frequency, fmax=self.config.max_frequency,
        )

        return librosa.power_to_db(melspec).astype(np.float32)
    
    def _mono_to_color(self, data, eps=1e-6, mean=None, std=None):
        mean = mean or data.mean()
        std = std or data.std()
        data = (data - mean) / (std + eps)
        
        _min, _max = data.min(), data.max()

        if (_max - _min) > eps:
            image = np.clip(data, _min, _max)
            image = 255 * (image - _min) / (_max - _min)
            image = image.astype(np.uint8)
        else:
            image = np.zeros_like(data, dtype=np.uint8)
            
        image = np.stack([image, image, image], axis=0)
        return image
    
    def _audio_to_image(self, audio):
        image = self._mono_to_color(audio)
        return torch.tensor(image, dtype=torch.float32)

    def read_data(self, row):
        path = os.path.join(self.path_to_data, row['filename'])
        path = path.split('.')[0]
        path += '.npy'

        audio = np.load(path)

        images = self._audio_to_image(audio)  
        labels = torch.tensor(list(row)[2:]).float()
        
        return images, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.read_data(self.data.loc[idx])