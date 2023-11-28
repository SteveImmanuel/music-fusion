import logging
import torch
import torch.nn.functional as F

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
        # gated = F.tanh(dilated_out[:, :dilated_out.shape[1] // 2, :]) * F.sigmoid(dilated_out[:, dilated_out.shape[1] // 2:, :])
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

    @property
    def receptive_field(self):
        return 1024

    def forward(self, x):
        logger.debug(f'input {x.shape}')
        x = self.start_conv(x)
        logger.debug(f'after start_conv {x.shape}')

        skip_connections = []
        for i in range(self.num_layers * self.num_stacks):
            x, skip = self.layers[i](x)
            logger.debug(f'residual {i} {x.shape}')
            logger.debug(f'skip {i} {skip.shape}')
            skip_connections.append(skip)

        out = torch.stack(skip_connections, dim=0)
        logger.debug(f'skip {out.shape}')
        out = torch.sum(out, dim=0)
        logger.debug(f'skip pool {out.shape}')
        
        out = F.relu(out)
        out = self.end_conv_1(out)
        out = F.relu(out)
        out = self.end_conv_2(out)
        # softmax is handled by the loss function
        logger.debug(f'output {out.shape}')
        out = out[:, :, -1]
        return out

    # def generate(self, x, num_samples):
    #     x = self.start_conv(x)
    #     skip_connections = []
    #     for i in range(self.num_layers * self.num_stacks):
    #         x, skip = self.layers[i](x)
    #         skip_connections.append(skip)
    #     x = sum(skip_connections)
    #     x = F.relu(x)
    #     x = self.end_conv_1(x)
    #     x = F.relu(x)
    #     x = self.end_conv_2(x)
    #     x = x[:, :, -1]
    #     x = F.softmax(x, dim=1)
    #     x = torch.multinomial(x, num_samples=num_samples)
    #     return x


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