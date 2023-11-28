import os
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import WaveNetPl, WaveNet
from data import AudioDataset
from tqdm import tqdm
from hyp import *

torch.set_float32_matmul_precision('high')

model = WaveNet(residual_channels=128, skip_channels=128, num_stacks=1)
pl_model = WaveNetPl(model)

# train_dataset = AudioDataset('dataset/train', receptive_field=model.receptive_field)
train_dataset = AudioDataset('dataset/val', receptive_field=model.receptive_field)
val_dataset = AudioDataset('dataset/val', receptive_field=model.receptive_field)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# early_stop_cb = EarlyStopping(monitor='val/batch_loss', patience=10, verbose=True, mode='min')
checkpoint_cb = ModelCheckpoint(every_n_train_steps=10000)
logger = TensorBoardLogger(save_dir=os.getcwd(), name='runs')
trainer = pl.Trainer(
    logger=logger,
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    default_root_dir=os.getcwd(),
    log_every_n_steps=5,
    accelerator='gpu',
    devices=2,
    callbacks=[checkpoint_cb],
    strategy='ddp_find_unused_parameters_true',
)
# trainer.fit(pl_model, train_dl, val_dl)
trainer.fit(pl_model, train_dl)