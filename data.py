import torch
import numpy as np
import librosa
import os

from nptyping import NDArray


def mu_law_encode(data: NDArray, mu: int = 255) -> NDArray:
    """
    Quantize waveform amplitudes.
    data range: [-1, 1]
    """
    data = np.sign(data) * (np.log(1 + mu * np.abs(data)) / np.log(1 + mu))
    data = np.floor((data + 1) / 2 * mu)
    return data.astype(np.int64)


def mu_law_decode(data: NDArray, mu: int = 255) -> NDArray:
    """
    Recovers waveform from quantized values.
    data range: [0, mu]
    """
    data = data.astype(np.float32)
    data = 2 * (data / mu) - 1
    data = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return data


class AudioDataset(torch.utils.data.Dataset):
    """
    |-- receptive field --|
    |------- samples -------------------|
    |---------------------|-- outputs --|
    """
    def __init__(self, root_dir: str, sample_rate: int = 16000, receptive_field: int = 1024, mu: int = 255):
        super().__init__()

        assert os.path.exists(root_dir), f"{root_dir} does not exist"

        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.receptive_field = receptive_field

        self._load_all_data()

    def _load_all_data(self):
        self.data_len = []
        self.data = []
        self.idx2lbl = {}

        lbl_idx = 0
        for label in os.listdir(self.root_dir):
            self.idx2lbl[lbl_idx] = label
            
            label_dir = os.path.join(self.root_dir, label)
            for file_name in os.listdir(label_dir):
                ext = file_name.split('.')[-1].lower()
                if ext not in ['wav']:
                    continue

                file_path = os.path.join(label_dir, file_name)
                wav, _ = librosa.load(file_path, sr=self.sample_rate)
                encoded_wav = mu_law_encode(wav)
                self.data.append((lbl_idx, encoded_wav))
                self.data_len.append(len(encoded_wav) - self.receptive_field)
            
            lbl_idx += 1


    def __len__(self):
        return sum(self.data_len)

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.data_len[file_idx]:
            idx -= self.data_len[file_idx]
            file_idx += 1

        lbl_idx = self.data[file_idx][0]
        wav_in = self.data[file_idx][1][idx:idx + self.receptive_field]
        wav_out = self.data[file_idx][1][idx + self.receptive_field]
        
        wav_in = torch.from_numpy(wav_in).long()
        wav_out = torch.tensor(wav_out).long()

        return lbl_idx, wav_in, wav_out


if __name__ == "__main__":
    # data = [0, -1, 1, -2**15, 2**15 - 1]
    # print(data)
    # data = quantize_data(data, 256)
    # # data = mu_law_decode(data)
    # print(data)
    dataset = AudioDataset('dataset')
    print(dataset[0][0], dataset[0][1], dataset[0][2])
    print(dataset[1][0], dataset[1][1], dataset[1][2])
