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
    {'freq': 261, 'dur': 0.7, 'amp': 0.5},   # C4
    {'freq': 293, 'dur': 1.2, 'amp': 0.5},   # D4
    {'freq': 349, 'dur': 0.6, 'amp': 0.5},   # F4
    {'freq': 392, 'dur': 1.4, 'amp': 0.5},   # G4
    {'freq': 493, 'dur': 0.9, 'amp': 0.5},   # B4
    {'freq': 523, 'dur': 1.1, 'amp': 0.5},   # C5
    {'freq': 587, 'dur': 0.8, 'amp': 0.5},   # D5
    {'freq': 659, 'dur': 1.3, 'amp': 0.5},   # E5
    {'freq': 698, 'dur': 0.5, 'amp': 0.5},   # F5
    {'freq': 784, 'dur': 1.0, 'amp': 0.5},   # G5
    {'freq': 250, 'dur': 1.5, 'amp': 0.5},   # Low
    {'freq': 370, 'dur': 0.6, 'amp': 0.5},   # Mid
    {'freq': 600, 'dur': 1.2, 'amp': 0.5},   # Mid-high
    {'freq': 750, 'dur': 0.7, 'amp': 0.5},   # High
    {'freq': 900, 'dur': 0.9, 'amp': 0.5},   # Very high
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

# NORMALIZE targets to 0-1 range for better learning
y_train[:, 0] = (y_train[:, 0] - 220) / (900 - 220)  # freq: 220-900 → 0-1
y_train[:, 1] = (y_train[:, 1] - 0.5) / (1.5 - 0.5)  # dur: 0.5-1.5 → 0-1  
y_train[:, 2] = y_train[:, 2] / 1.0                   # amp: already 0-1

print(f"Loaded {len(y_train)} training samples (normalized to 0-1)")

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset)

def get_data():
    return train_loader