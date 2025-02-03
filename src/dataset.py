import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm.auto import tqdm

from .utils import PHONEMES


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, config, phonemes=PHONEMES, partition="train-clean-100"):
        self.context = config["context"]
        self.phonemes = phonemes
        self.subset = config["subset"]

        self.aug_prob = config["aug_prob"]
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=config["freq_mask_param"])
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=config["time_mask_param"])

        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")

        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))

        subset_size = int(self.subset * len(mfcc_names))

        mfcc_names = mfcc_names[:subset_size]
        transcript_names = transcript_names[:subset_size]

        assert len(mfcc_names) == len(transcript_names)

        self.mfccs = []
        self.transcripts = []
        self.original_lengths = []  # Track T_i (original length per utterance)
        self.padded_lengths = []    # Track T_i + 2*K (padded length per utterance)
        self.start_indices = [0]    # Track start indices of each padded utterance

        for i in tqdm(range(len(mfcc_names))):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0))

            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]))[1:-1]

            phoneme2idx = {p: i for i, p in enumerate(self.phonemes)}
            transcript_indices = np.array([phoneme2idx[p] for p in transcript])

            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)

            self.mfccs.append(mfccs_normalized)
            self.transcripts.append(transcript_indices)

        self.mfccs = torch.cat(self.mfccs, dim=0)
        self.transcripts = torch.cat(self.transcripts, dim=0)

        self.length = len(self.mfccs)
        self.mfccs = nn.functional.pad(self.mfccs, (0, 0, self.context, self.context), mode="constant", value=0)

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)

        if torch.rand(1) < self.aug_prob:
            x = x.transpose(1, 2)  # Shape: (batch_size, freq, time)
            x = self.freq_masking(x)
            x = self.time_masking(x)
            x = x.transpose(1, 2)  # Shape back to: (batch_size, time, freq)

        return x, torch.tensor(y)

    def __getitem__(self, ind):
        padded_ind = ind + self.context
        frames = self.mfccs[padded_ind - self.context : padded_ind + self.context + 1]
        phonemes = self.transcripts[ind]
        return frames.contiguous(), phonemes


class AudioTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, config, partition="test-clean"):
        self.context = config["context"]

        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.mfccs = []
        self.original_lengths = []
        self.start_indices = [0]

        for mfcc_name in tqdm(self.mfcc_names):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_name))
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0))
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            # Pad each utterance
            padded_mfcc = nn.functional.pad(
                mfccs_normalized,
                (0, 0, self.context, self.context), 
                mode="constant", value=0
            )
            self.mfccs.append(padded_mfcc)
            self.original_lengths.append(len(mfcc))
            self.start_indices.append(self.start_indices[-1] + len(padded_mfcc))

        self.mfccs = torch.cat(self.mfccs, dim=0)

        self.length = len(self.mfccs)

        self.mfccs = nn.functional.pad(self.mfccs, pad=(0, 0, self.context, self.context), mode="constant", value=0)


    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)

    def __getitem__(self, ind):
        utterance_idx = np.searchsorted(self.cumulative_original, ind, side='right')
        if utterance_idx > 0:
            local_ind = ind - self.cumulative_original[utterance_idx - 1]
        else:
            local_ind = ind

        start_pos = self.start_indices[utterance_idx]
        window_start = start_pos + local_ind
        window_end = window_start + 2 * self.context + 1
        frames = self.mfccs[window_start:window_end]
        return frames


def create_dataloaders(config, root):
    train_data = AudioDataset(root=root, config=config, partition="train-clean-100")
    val_data = AudioDataset(root=root, config=config, partition="dev-clean")
    test_data = AudioTestDataset(root=root, config=config, partition="test-clean")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        num_workers=min(8, os.cpu_count()),
        batch_size=config["batch_size"],
        pin_memory=True,
        shuffle=True,
        collate_fn=train_data.collate_fn,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        num_workers=0,
        batch_size=config["batch_size"],
        pin_memory=True,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=0,
        batch_size=config["batch_size"],
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
