import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import soundfile
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import WaveNetPl, WaveNet
from data import AudioDataset
from tqdm import tqdm
from hyp import *
from data import mu_law_decode

torch.set_float32_matmul_precision('high')

model = WaveNet(residual_channels=128, skip_channels=128, num_stacks=1)
pl_model = WaveNetPl.load_from_checkpoint('runs/version_12/checkpoints/epoch=5-step=10000.ckpt', model=model)
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
    soundfile.write('deep_fusion_2.wav', result, flute_dataset.sample_rate)




    # result = []
    # eval_dataset.set_override_file_idx(0)
    # dl = torch.utils.data.DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    # # flute
    # for lbl_idx, wav_in, wav_out in tqdm(dl):
    #     batch_size = len(lbl_idx)
    #     # lbl_idx = torch.ones(batch_size, dtype=torch.long)
    #     # lbl_idx = lbl_idx.to(DEVICE)
    #     wav_in = wav_in.to(DEVICE)
    #     wav_out = wav_out.to(DEVICE)
    #     pred = pl_model.model.forward_fusion(wav_in)
    #     pred = torch.argmax(pred, dim=1)
    #     pred = pred.cpu().detach().numpy()
    #     result.append(pred)

    # result = np.concatenate(result)
    # result = mu_law_decode(result)
    # soundfile.write('flute_with_fusion_embedding.wav', result, eval_dataset.sample_rate)

    # result = []
    # eval_dataset.set_override_file_idx(1)
    # dl = torch.utils.data.DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    # # flute
    # for lbl_idx, wav_in, wav_out in tqdm(dl):
    #     batch_size = len(lbl_idx)
    #     # lbl_idx = torch.zeros(batch_size, dtype=torch.long)
    #     # lbl_idx = lbl_idx.to(DEVICE)
    #     wav_in = wav_in.to(DEVICE)
    #     wav_out = wav_out.to(DEVICE)
    #     pred = pl_model.model.forward_fusion(wav_in)
    #     pred = torch.argmax(pred, dim=1)
    #     pred = pred.cpu().detach().numpy()
    #     result.append(pred)

    # result = np.concatenate(result)
    # result = mu_law_decode(result)
    # soundfile.write('violin_with_fusion_embedding.wav', result, eval_dataset.sample_rate)
