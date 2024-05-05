import math
import os
import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config

def create_and_save_melspec(train_folders: list[str], path_to_save: str, augmenation=None):
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

    config = Config()

    for i, data_folder in enumerate(train_folders):
        species = os.listdir(data_folder)
        print('Current folder: {}'.format(i))

        for bird in tqdm(species):

            # Check if it's a new class
            is_new_class = False
            new_dir = os.path.join(path_to_save, bird)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
                is_new_class = True

            files = os.listdir(os.path.join(data_folder, bird))

            for file in files:
                path_to_new_file = os.path.join(path_to_save, bird, file[:-4])
               
                try:
                    y, sr = librosa.load(os.path.join(data_folder, bird, file), sr=config.sample_rate)
            
                    # Repeat audio if its length is less than the window length
                    n_copy = math.ceil(config.max_time * config.sample_rate / len(y))
                    if n_copy > 1:
                        y = np.concatenate([y] * n_copy)

                    # Extract a fixed-length window from the middle of the audio
                    start_idx = int(len(y) / 2 - 2.5 * config.sample_rate)
                    end_idx = int(start_idx + config.max_time * config.sample_rate)
                    y = y[start_idx:end_idx]

                    melspec = librosa.feature.melspectrogram(
                        y=y, 
                        sr=sr, 
                        fmin=config.min_frequency, 
                        fmax=config.max_frequency, 
                        n_mels=config.n_mels, 
                        n_fft=config.n_fft)
                    
                    melspec = librosa.power_to_db(melspec).astype(np.float32)

                    path_to_new_file = os.path.join(path_to_save, bird, file[:-4])

                    melspec = cv2.resize(melspec, (256, 256), interpolation=cv2.INTER_AREA)

                    # Skip saving if file already exists
                    # if is_new_class or not os.path.exists(path_to_new_file):
                    np.save(path_to_new_file, melspec)
                    
                except:
                    print('Error while processing {}'.format(os.path.join(data_folder, bird, file)))
                    continue
                                                                

def get_filtered_data(this_year_metadata: str, old_metadata: list[str], to_csv: bool=True) -> pd.DataFrame:
    """
    Gets all files with metadata and filter old datasets by next rule:
    1. Remove all files with equal name, author and label.
    2. Remove all species with less than 20 elements.

    Concatenate all data together
    """
    old_data = pd.DataFrame()
    for file in old_metadata:
        old_data = pd.concat([old_data, pd.read_csv(file)])

    print('Total number of elements: {}'.format(len(old_data)))
    old_data = old_data.drop_duplicates(['filename', 'primary_label', 'author'])
    old_data = old_data.groupby('primary_label').filter(lambda group: len(group) >= 20)
    print('Number of elements in filtered data: {}'.format(len(old_data)))

    new_data = pd.read_csv(this_year_metadata)
    data = pd.concat([new_data, old_data])

    if to_csv:
        new_data.to_csv('this_year_only_metadata.csv', index=False)
        old_data.to_csv('past_years_metadata.csv', index=False)
        data.to_csv('full_dataset_metadata.csv', index=False)

    return data

def get_species_with_low_elements(data: pd.DataFrame) -> list[str]:
    """
    """

    return data.groupby('primary_label').filter(lambda group: len(group) < 20)['primary_label'].unique().tolist()


if __name__ == '__main__':
    # filtered_data = get_filtered_data('/home/asphodel/Downloads/birdclef-2024/train_metadata.csv', 
    #                 ['/home/asphodel/Downloads/birdclef-2021/train_metadata.csv', 
    #                 '/home/asphodel/Downloads/birdclef-2022/train_metadata.csv',
    #                 '/home/asphodel/Downloads/birdclef-2023/train_metadata.csv'])
    
    # species_to_augment = get_species_with_low_elements(filtered_data)

    create_and_save_melspec(['/home/asphodel/Downloads/birdclef-2021/train_short_audio'],
                             '/home/asphodel/Code/ml-dl-env/birdclef2024-competition/melspecs')



    
    




    
    

