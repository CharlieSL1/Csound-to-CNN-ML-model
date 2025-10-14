'''
If you change the csound_module.py, run this cell to reload
reload_module()

'''
from sympy.external import importtools
from csound_module import *
import librosa # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

audio_folder = 'dataset'
csound = Csound_instrument()
X_train = []  
y_train = []

params_list = [
    {'freq': 220, 'dur': 1.0, 'amp': 0.5},   # A3
    {'freq': 440, 'dur': 1.0, 'amp': 0.5},   # A4
    {'freq': 880, 'dur': 0.5, 'amp': 0.5},   # A5
    {'freq': 330, 'dur': 1.5, 'amp': 0.5},   # E4
    {'freq': 550, 'dur': 0.8, 'amp': 0.5},   # C#5
]

for i, params in enumerate(params_list):
    filename = f"dataset/train_{i}.wav"
    csound.csound_instrument_score(
        instr_number=1,
        frequency=params['freq'],
        duration=params['dur'],
        save_file=filename
    )
    
    audio, sr = librosa.load(filename, sr=48000)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    
    max_len = 48000 * 2  # 2秒
    if audio.shape[1] < max_len:
        audio = np.pad(audio, ((0, 0), (0, max_len - audio.shape[1])))
    else:
        audio = audio[:, :max_len]
    
    X_train.append(audio)
    y_train.append([params['freq'], params['dur'], params['amp']])

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset)

def get_data():
    return train_loader