import pandas as pd


class SpectrogramConfig:
    def __init__(self):
        # Spectrogram calculation parameters
        self.n_fft = 1024
        self.num_fold = 5
        self.window_duration_in_sec = 5
        self.n_mels = 128 

class DataConfig:
    def __init__(self):
        # Data parameters
        self.max_time = 5
        self.sample_rate = 32000
        self.audio_length = self.max_time * self.sample_rate
        self.min_frequency = 60
        self.max_frequency = 16000

class ModelConfig:
    def __init__(self):
        # Model parameters
        self.batch_size = 32
        self.inference_chunks_number = 48
        self.epochs = 30
        self.learning_rate = 1e-3
        self.num_classes = 182

class OptimizerConfig:
    def __init__(self):
        # Optimizer parameters
        self.eta_min = 1e-6

class Config(SpectrogramConfig, DataConfig, ModelConfig, OptimizerConfig):
    def __init__(self):
        SpectrogramConfig.__init__(self)
        DataConfig.__init__(self)
        ModelConfig.__init__(self)
        OptimizerConfig.__init__(self)

        self.path_to_data = '/home/asphodel/Downloads/melspecs'
        self.metadata = '/home/asphodel/Code/dl-env/birdclef2024-competition/this_year_only_metadata.csv'
        self.labels = list(pd.read_csv(self.metadata)['primary_label'].unique())
