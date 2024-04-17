import torch

class Config:
    def __init__(self):
        # Spectrogram calculation parameters
        self.nfft = 1024
        self.num_fold = 5
        self.window_duration_in_sec = 5
        self.n_mels = 128 
        self.width = 256
        
        # Data parameters
        self.max_time = 5
        self.sample_rate = 32000
        self.audio_length = self.max_time * self.sample_rate
        self.min_frequency = 0
        self.max_frequency = 16000

        # Model parameters
        self.model_name = 'tf_efficientnetv2_s.in21k'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 5e-4
        self.num_classes = 182

        self.metadata = '/home/asphodel/Code/ml-dl-env/birdclef2024-competition/labels-2024.csv'