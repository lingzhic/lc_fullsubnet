import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile


class Edingburgh(Dataset):
    """Dataset class for Edingburgh by LC

    Args:
        csv_path (str): The path to the metadata file.
        target_sr (int) : sample rate of the clean and noisys (48000 for the orignal Edingburgh dataset).
        segment (int, optional) : The desired clean and noisys length in s.
    """

    dataset_name = "Edingburgh"

    def __init__(self, csv_path, target_sr=8000, segment=3, return_id=False):
        super().__init__()
        self.return_id = return_id
        # Get the meta csv
        self.csv_path = csv_path
        # md_file = [f for f in os.listdir(csv_dir)][0]
        # self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = 48000
        self.target_sr = target_sr
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["Length (s)"]*self.sample_rate >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row of data at given index in dataframe
        row = self.df.iloc[idx]

        # NOTE If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, int(row["Length (s)"]*self.sample_rate - self.seg_len))
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        
        # Get noisy path
        noisy_path = row["Noisy_filepath"]
        self.noisy_path = noisy_path

        # Get the clean path
        clean_list = []
        clean_path = row[f"Clean_filepath"]
        s, _ = sf.read(clean_path, dtype="float32", start=start, stop=stop)
        s_tar = librosa.resample(s, orig_sr=self.sample_rate, target_sr=self.target_sr)         # after loading file from start to stop, resample it to 8k sample rate
        clean_list.append(s_tar)

        # Read the noisy
        noisy, _ = sf.read(noisy_path, dtype="float32", start=start, stop=stop)
        noisy_tar = librosa.resample(noisy, orig_sr=self.sample_rate, target_sr=self.target_sr)
        noisy_tar = torch.from_numpy(noisy_tar)         # Convert to torch tensor
        # Stack clean, convert clean to tensor
        clean_tar = np.vstack(clean_list)               # Here we only have one source of clean speech, so the stack only have 1 row
        clean_tar = torch.from_numpy(clean_tar)
        if not self.return_id:
            return noisy_tar, clean_tar
        
        id1, id2 = noisy_path.split("/")[-1].split(".")[0].split("_")
        return noisy_tar, clean_tar, [id1, id2]
    
    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        return infos
