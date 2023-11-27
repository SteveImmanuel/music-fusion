import torch
import torch.nn.functional as F


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

        self.dilated_conv = CausalConv1d(residual_channels, 2 * residual_channels, kernel_size, dilation=dilation)
        self.residual = torch.nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip = torch.nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        dilated_out = self.dilated_conv(x)
        # gated = F.tanh(dilated_out[:, :dilated_out.shape[1] // 2, :]) * F.sigmoid(dilated_out[:, dilated_out.shape[1] // 2:, :])
        gated = F.tanh(dilated_out) * F.sigmoid(dilated_out)
        residual_out = self.residual(gated)
        skip_out = self.skip(gated)
        return x + residual_out, skip_out