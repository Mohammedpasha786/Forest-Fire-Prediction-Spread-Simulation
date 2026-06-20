Auxiliary temporal model. While the U-NET predicts fire risk from a single
day's spatial feature stack, fire danger also depends on antecedent weather
(e.g. a multi-day dry spell matters more than one hot day). This ConvLSTM
consumes a short sequence of past daily weather rasters and produces a
"temporal danger" feature map that is concatenated into the U-NET input
stack (see src/prediction/train_unet.py --use_temporal_context flag).

This keeps the temporal and spatial models decoupled and independently
testable, while still satisfying the "U-NET or LSTM" suggested-tools
guidance with both implemented.

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class FireDangerConvLSTM(nn.Module):
    """
    Input:  (N, T, C_weather, H, W)  - T days of weather rasters
            (temperature, rh, wind_speed, wind_dir_sin, wind_dir_cos, rainfall)
    Output: (N, 1, H, W) - learned temporal danger scalar field in [0, 1],
            intended to be concatenated as an extra channel before the U-NET.
    """

    def __init__(self, in_channels=6, hidden_dim=16, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_dim, kernel_size)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden_dim, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, t, c, h, w = x.shape
        h_t, c_t = self.cell.init_hidden(n, h, w, x.device)
        for step in range(t):
            h_t, c_t = self.cell(x[:, step], h_t, c_t)
        return self.readout(h_t)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FireDangerConvLSTM(in_channels=6, hidden_dim=16)
    dummy = torch.randn(2, 7, 6, 128, 128)  # 7-day sequence
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
