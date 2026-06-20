U-NET architecture for pixel-wise forest fire risk classification.
Input:  (N, C, H, W) multi-channel feature patches
Output: (N, num_classes, H, W) logits -> argmax gives risk class per pixel
        Classes: 0=nil/very_less, 1=low/less, 2=moderate, 3=high/very_high

Default in_channels=13 matches the feature_stack.py manifest:
    lulc_norm, slope, aspect_sin, aspect_cos, ndvi, fuel_load, temperature,
    rh, wind_speed, wind_dir_sin, wind_dir_cos, rainfall, dist_to_road

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FireRiskUNet(nn.Module):
    """
    Standard 4-level U-NET adapted for multi-channel geospatial input.
    base_filters controls model capacity; depth=4 -> 4 down/up stages (matches
    the encoder/decoder depicted in the architecture diagram, docs/diagrams/).
    """

    def __init__(self, in_channels=13, num_classes=4, base_filters=32, depth=4):
        super().__init__()
        self.depth = depth
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.in_conv = DoubleConv(in_channels, filters[0])
        self.downs = nn.ModuleList([Down(filters[i], filters[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([Up(filters[i + 1], filters[i]) for i in reversed(range(depth))])
        self.out_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = [self.in_conv(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)

        return self.out_conv(x)

    @torch.no_grad()
    def predict_risk_map(self, x):
        """Returns (N, H, W) class indices and (N, num_classes, H, W) softmax probs."""
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(probs, dim=1)
        return classes, probs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FireRiskUNet(in_channels=13, num_classes=4, base_filters=32, depth=4)
    dummy = torch.randn(2, 13, 256, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
