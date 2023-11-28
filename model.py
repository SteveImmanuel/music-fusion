import logging
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from hyp import *

logger = logging.getLogger('Model')
logging.basicConfig(level=logging.INFO)


class CausalConv1d(torch.nn.Conv1d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1, **kwargs):
        padding = dilation * (kernel_size - 1)
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        return out[:, :, :-self.padding[0]]


class WaveNetLayer(torch.nn.Module):

    def __init__(self, residual_channels: int, skip_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()

        self.dilated_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation=dilation)
        self.residual = torch.nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip = torch.nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        dilated_out = self.dilated_conv(x)
        gated = F.tanh(dilated_out) * F.sigmoid(dilated_out)
        residual_out = self.residual(gated)
        skip_out = self.skip(gated)
        return x + residual_out, skip_out


class WaveNet(torch.nn.Module):

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        num_classes: int = 256,
        num_layers: int = 10,
        num_stacks: int = 3,
        kernel_size: int = 2,
    ):
        super().__init__()

        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.start_conv = CausalConv1d(1, residual_channels, kernel_size=kernel_size)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_stacks):
            for i in range(num_layers):
                dilation = 2**i
                self.layers.append(WaveNetLayer(residual_channels, skip_channels, kernel_size, dilation))
        self.end_conv_1 = torch.nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.end_conv_2 = torch.nn.Conv1d(skip_channels, num_classes, kernel_size=1)

        self.bn1 = torch.nn.BatchNorm1d(skip_channels)
        self.bn2 = torch.nn.BatchNorm1d(skip_channels)
        self.embedding = torch.nn.Embedding(2, self.receptive_field)

    @property
    def receptive_field(self):
        return 1024

    def forward(self, lbl_idx, wav_in):
        x = self.start_conv(wav_in)
        latent = self.embedding(lbl_idx).unsqueeze(1).expand(-1, self.skip_channels, -1)

        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_connections.append(skip)

        out = torch.stack(skip_connections, dim=0)
        out = torch.sum(out, dim=0)

        out = F.relu(out)

        out = out + latent
        out = self.bn1(out)
        out = self.end_conv_1(out)
        out = F.relu(out)

        out = self.bn2(out)
        out = self.end_conv_2(out)
        # softmax is handled by the loss function
        out = out[:, :, -1]
        return out
    
    def forward_fusion(self, wav_in):
        batch_size = len(wav_in)
        lbl_idx_ones = torch.ones(batch_size, dtype=torch.long).to(wav_in.device)
        lbl_idx_zeros = torch.zeros(batch_size, dtype=torch.long).to(wav_in.device)
        x = self.start_conv(wav_in)
        latent_zero = self.embedding(lbl_idx_zeros).unsqueeze(1).expand(-1, self.skip_channels, -1)
        latent_one = self.embedding(lbl_idx_ones).unsqueeze(1).expand(-1, self.skip_channels, -1)
        latent = latent_zero + latent_one
        # latent = latent / 2

        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_connections.append(skip)

        out = torch.stack(skip_connections, dim=0)
        out = torch.sum(out, dim=0)

        out = F.relu(out)

        out = out + latent
        out = self.bn1(out)
        out = self.end_conv_1(out)
        out = F.relu(out)

        out = self.bn2(out)
        out = self.end_conv_2(out)
        # softmax is handled by the loss function
        out = out[:, :, -1]
        return out
    
    def forward_deep_fusion(self, wav1_in, wav2_in):
        batch_size = len(wav1_in)
        lbl_idx_ones = torch.ones(batch_size, dtype=torch.long).to(wav1_in.device)
        lbl_idx_zeros = torch.zeros(batch_size, dtype=torch.long).to(wav1_in.device)
        latent_zero = self.embedding(lbl_idx_zeros).unsqueeze(1).expand(-1, self.skip_channels, -1)
        latent_one = self.embedding(lbl_idx_ones).unsqueeze(1).expand(-1, self.skip_channels, -1)
        latent = latent_zero + latent_one

        x1 = self.start_conv(wav1_in)
        x2 = self.start_conv(wav2_in)

        skip_connections = []
        skip_connections1 = []
        skip_connections2 = []
        for layer in self.layers:
            x1, skip1 = layer(x1)
            skip_connections1.append(skip1)

            x2, skip2 = layer(x2)
            skip_connections2.append(skip2)

            x = x1 + x2
            x, skip = layer(x)
            skip_connections.append(skip)

        out = torch.stack(skip_connections + skip_connections1 + skip_connections2, dim=0)
        out = torch.sum(out, dim=0)

        out = F.relu(out)

        out = out + latent
        out = self.bn1(out)
        out = self.end_conv_1(out)
        out = F.relu(out)

        out = self.bn2(out)
        out = self.end_conv_2(out)
        # softmax is handled by the loss function
        out = out[:, :, -1]
        return out

class WaveNetPl(pl.LightningModule):

    def __init__(self, model: WaveNet):
        super().__init__()
        self.model = model
        # self.training_step_outputs = []
        # self.validation_step_outputs = []

    def _calculate_acc(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        return torch.sum(y_hat == y).item() / len(y)

    def _calculate_acc_raw(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        return torch.sum(y_hat == y).item(), len(y)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        self.model.train()
        lbl_idx, wav_in, wav_out = batch
        pred = self.model(lbl_idx, wav_in)
        loss = F.cross_entropy(pred, wav_out)
        acc = self._calculate_acc(pred, wav_out)
        self.log('train/batch_loss', loss, sync_dist=True)
        self.log('train/batch_acc', acc, sync_dist=True)
        # self.training_step_outputs.append({'loss': loss, 'pred': pred, 'target': wav_out})
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        self.model.eval()
        lbl_idx, wav_in, wav_out = batch
        with torch.no_grad():
            pred = self.model(lbl_idx, wav_in)
        loss = F.cross_entropy(pred, wav_out)
        acc = self._calculate_acc(pred, wav_out)
        self.log('val/batch_loss', loss, sync_dist=True)
        self.log('val/batch_acc', acc, sync_dist=True)
        # self.validation_step_outputs.append({'loss': loss, 'pred': pred, 'target': wav_out})
        return loss

    # def on_train_epoch_end(self) -> None:
    #     outputs = self.training_step_outputs
    #     total_correct = 0
    #     total_data = 0
    #     total_loss = 0
    #     for output in outputs:
    #         b_correct, b_data = self._calculate_acc_raw(output['pred'], output['target'])
    #         total_correct += b_correct
    #         total_data += b_data
    #         total_loss += output['loss']

    #     self.log('train/acc', total_correct / total_data, prog_bar=True, sync_dist=True)
    #     self.log('train/loss', total_loss / len(outputs), prog_bar=True, sync_dist=True)
    #     self.training_step_outputs.clear()

    # def on_validation_epoch_end(self) -> None:
    #     outputs = self.validation_step_outputs
    #     total_correct = 0
    #     total_data = 0
    #     total_loss = 0
    #     for output in outputs:
    #         b_correct, b_data = self._calculate_acc_raw(output['pred'], output['target'])
    #         total_correct += b_correct
    #         total_data += b_data
    #         total_loss += output['loss']

    #     self.log('val/acc', total_correct / total_data, prog_bar=True, sync_dist=True)
    #     self.log('val/loss', total_loss / len(outputs), prog_bar=True, sync_dist=True)
    #     self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler':
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=LR_DECAY_FACTOR,
                    patience=LR_DECAY_PATIENCE,
                    min_lr=LR_MIN,
                ),
                'monitor':
                'train/batch_loss',
                'frequency':
                1,
            },
        }


if __name__ == '__main__':
    # c1d = CausalConv1d(1, 3, dilation=1, kernel_size=2)
    # x = torch.randn(1, 1, 10)
    # print(x)
    # print(x.shape)
    # print(c1d(x))
    # print(c1d(x).shape)
    wavenet = WaveNet(32, 32)
    x = torch.randn(1, 1, 20)
    print(wavenet(x).shape)
