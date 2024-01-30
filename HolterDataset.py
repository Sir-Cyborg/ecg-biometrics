import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, iirnotch
import os
import random

def filter_signal(signal, fs=128):
    [b, a] = butter(3, (0.5, 40), btype='bandpass', fs=fs)
    signal = filtfilt(b, a, signal, axis=0)
    [bn, an] = iirnotch(50, 3, fs=fs)
    signal = filtfilt(bn, an, signal, axis=0)
    return signal


# This fetch data from dir then store the filtered ecgs in .npy file
def load(cache_path):
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    signals_file = os.path.join(cache_path, 'ecg_signals.npy')
    ids_file = os.path.join(cache_path, 'ids.npy')

    if os.path.exists(signals_file) and os.path.exists(ids_file):
        ecg_signals = np.load(signals_file)
        ids = np.load(ids_file)
    else:
        ecg_signals = []
        ids = []
        for filename in os.listdir('files/'):
            nome_base, _ = os.path.splitext(filename)
            if nome_base not in ids:
                ids.append(nome_base)

                record = wfdb.rdrecord(f"{'files/'}/{nome_base}")

                start_point = 128*15*60
                end_point = 128*10*3600

                ecg_signal = record.p_signal[start_point:end_point, [0,1]] # ecg start 15 min to 10 h

                assert record.fs == 128, 'Sampling rate is not 128'

                idx, _ = np.where(np.isnan(ecg_signal))

                xx = np.arange(0, len(ecg_signal))
                xx_valid = np.setdiff1d(xx, idx)

                for channel in range(ecg_signal.shape[1]):
                    ecg_signal[:, channel] = np.interp(xx, xx_valid, ecg_signal[xx_valid, channel])
                
                ecg_signal = filter_signal(ecg_signal, fs=128)
                
                ecg_signals.append(ecg_signal)
            
        ecg_signals = np.array(ecg_signals)
        ecg_signals = np.swapaxes(ecg_signals, 1, 2)
        ids = np.array(ids)

        np.save(signals_file, ecg_signals)
        np.save(ids_file, ids)
    return ecg_signals, ids

class ECGDataset(Dataset):
    def __init__(self, ecgs, ids, fs, n_windows, seconds):
        self.ecg_signals = ecgs
        self.ids = ids
        self.id_mapped = {int(k):v for v, k in enumerate(self.ids)} # map the ids in integers
        self.id_mapped_tensor = torch.tensor([self.id_mapped[int(x)] for x in self.ids]) # then it convert in tensors

        self.fs = fs
        self.seconds = seconds
        self.n_windows = n_windows

        self.cut_ecg, self.cut_id, self.cut_id_mapped= self.cut()
        self.cut_id_mapped_tensor = torch.tensor(self.cut_id_mapped)
        self.classes = list(set(self.cut_id_mapped))
    
    def cut(self):
        sig = []
        id = []
        id_mapped = []
        i = 0
        window_size = int(self.seconds*self.fs)
        n_windows = self.n_windows
        N = self.ecg_signals.shape[2]
        for signal in self.ecg_signals:
            random_idx = [random.randint(0, N - window_size - 1) for i in range(n_windows)]
            random_idx.sort()
            for w in range(n_windows):
                start_point = random_idx[w]
                end_point = start_point + window_size 
                sig.append(signal[:,start_point:end_point])
                id.append(self.ids[i])
                id_mapped.append(self.id_mapped_tensor[i])
            i += 1   

        sig = np.array(sig)
        id = np.array(id)
        id_mapped = np.array(id_mapped)
        return sig, id, id_mapped

    def __len__(self):
        return len(self.cut_ecg)

    def __getitem__(self, index):
        ecg_signal = self.cut_ecg[index, :, :]
        patient_id = self.cut_id[index]
        label_class = self.cut_id_mapped_tensor[index]
        return torch.tensor(ecg_signal).type(torch.float), patient_id, label_class

