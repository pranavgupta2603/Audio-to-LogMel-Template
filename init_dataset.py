from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import pandas as pd
import librosa
import torch
import numpy as np
import cv2

class UrbanSound8KDataset(Dataset):
    def __init__(self, dataframe, fold=None, val=False, test=False):
        self.fold = fold  
        all_folds = [i for i in range(1, 11)]
        test_fold = 10-fold
        all_folds.remove(test_fold)
        if test==False:
            df = dataframe[dataframe['fold'].isin(all_folds)]
            train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
            if val ==True:
                self.dataframe = val_df
            else:
                self.dataframe = train_df
               
        elif test==True:
            self.dataframe = dataframe[dataframe['fold'] == test_fold]
    def __getitem__(self, index):
        path_to_file = self.get_path_to_file(index)
        signal = self.preprocess_signal(path_to_file)

        x = np.stack([cv2.resize(signal, (224, 224)) for _ in range(3)])

        y = self.dataframe.classID.values[index]
        return torch.tensor(x, dtype=torch.float), y

    def get_path_to_file(self, index):
        return f'urbansound8k/fold{self.dataframe.fold.values[index]}/{self.dataframe.slice_file_name.values[index]}'
    def preprocess_signal(self, path_to_file):
        signal, _ = librosa.load(path_to_file, sr=16000)
        signal = librosa.feature.melspectrogram(y=signal)
        signal = librosa.power_to_db(signal)
        return signal

    def __len__(self):
        return self.dataframe.shape[0]

class ESCDataset(Dataset):
    def __init__(self, dataframe, fold=None, val=False, test=False):
        
        self.fold = fold  
        all_folds = [1, 2, 3, 4, 5]
        test_fold = 5-fold
        all_folds.remove(test_fold)
        if test==False:
            df = dataframe[dataframe['fold'].isin(all_folds)]
            train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
            if val ==True:
                self.dataframe = val_df
            else:
                self.dataframe = train_df
               
        elif test==True:
            self.dataframe = dataframe[dataframe['fold'] == test_fold]
    def __getitem__(self, index):
        path_to_file = self.get_path_to_file(index)
        signal = self.preprocess_signal(path_to_file)

        x = np.stack([cv2.resize(signal, (224, 224)) for _ in range(3)])
        y = self.dataframe.target.values[index]
        return torch.tensor(x, dtype=torch.float), int(y)
    
    def get_path_to_file(self, index):
        return f'/environmental-sound-classification-50//audio/audio/16000/{self.dataframe.filename.values[index]}'
    def preprocess_signal(self, path_to_file):
        signal, _ = librosa.load(path_to_file, sr=16000)
        signal = librosa.feature.melspectrogram(y=signal)
        signal = librosa.power_to_db(signal)
        return signal

    def __len__(self):
        return self.dataframe.shape[0]