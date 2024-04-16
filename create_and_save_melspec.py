import math
import os
import librosa
import numpy as np
from tqdm import tqdm

def create_and_save_melspec(train_folders: list[str], 
                            path_to_save: str, 
                            n_fft=2048, 
                            n_mels=128, 
                            min_freq=40, 
                            max_freq=15000, 
                            sample_rate=32000, 
                            window_in_seconds=5):
    """
    Create and save mel spectrograms for audio files in the specified folders.

    Args:
        train_folders (list[str]): List of paths to folders containing audio data.
        path_to_save (str): Path to save the mel spectrograms.
        n_fft (int): Number of FFT points.
        n_mels (int): Number of mel bands to generate.
        min_freq (int): Minimum frequency in Hz.
        max_freq (int): Maximum frequency in Hz.
        sample_rate (int): Sampling rate of audio files.
        window_in_seconds (int): Length of the window in seconds.

    Returns:
        None
    """

    for i, data_folder in enumerate(train_folders):
        species = os.listdir(data_folder)
        print('Current folder: {}'.format(i))

        for bird in tqdm(species):

            # Check if it's a new class
            is_new_class = False
            new_dir = os.path.join(path_to_save, bird)
            if os.path.exists(new_dir) == False:
                os.mkdir(new_dir)
                is_new_class = True

            files = os.listdir(os.path.join(data_folder, bird))
       
            for file in files:
                y, sr = librosa.load(os.path.join(data_folder, bird, file), sr=sample_rate)

                # Repeat audio if its length is less than the window length
                n_copy = math.ceil(window_in_seconds * sample_rate / len(y))
                if n_copy > 1: 
                    y = np.concatenate([y] * n_copy)

                # Extract a fixed-length window from the middle of the audio
                start_idx = int(len(y) / 2 - 2.5 * sample_rate)
                end_idx = int(start_idx + 5.0 * sample_rate)
                y = y[start_idx:end_idx]

                melspec = librosa.feature.melspectrogram(y=y, sr=sr, fmin=min_freq, fmax=max_freq, n_mels=n_mels, n_fft=n_fft)
                melspec = librosa.power_to_db(melspec).astype(np.float32)

                path_to_new_file = os.path.join(path_to_save, bird, file[:-4])

                # Skip saving if file already exists
                if is_new_class or os.path.exists(path_to_new_file) == False:
                    np.save(path_to_new_file, melspec)

create_and_save_melspec(['/home/asphodel/Downloads/birdclef-2024/train_audio', 
                         '/home/asphodel/Downloads/birdclef-2023/train_audio', 
                         '/home/asphodel/Downloads/birdclef-2022/train_audio'],
                        '/home/asphodel/Code/ml-dl-env/birdclef2024-competition/melspecs')
