import torch
import numpy as np
import soundfile
import argparse
from model import WaveNetPl, WaveNet
from data import AudioDataset
from tqdm import tqdm
from hyp import *
from data import mu_law_decode

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

model = WaveNet(residual_channels=128, skip_channels=128, num_stacks=3)
pl_model = WaveNetPl.load_from_checkpoint(args.model, model=model)
pl_model.to(DEVICE)

with torch.inference_mode():
    flute_dataset = AudioDataset('dataset/val', receptive_field=model.receptive_field)
    flute_dataset.set_override_file_idx(0)
    flute_dl = torch.utils.data.DataLoader(flute_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    violin_dataset = AudioDataset('dataset/val', receptive_field=model.receptive_field)
    violin_dataset.set_override_file_idx(1)
    violin_dl = torch.utils.data.DataLoader(violin_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

    result = []
    for (_, wav_in1, _), (_, wav_in2, _) in tqdm(zip(flute_dl, violin_dl)):
        wav_in1 = wav_in1.to(DEVICE)
        wav_in2 = wav_in2.to(DEVICE)
        pred = pl_model.model.forward_deep_fusion(wav_in1, wav_in2)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().detach().numpy()
        result.append(pred)
    
    result = np.concatenate(result)
    result = mu_law_decode(result)
    soundfile.write('deep_fusion.wav', result, flute_dataset.sample_rate)