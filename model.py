import torch
from torch import nn
import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DiffusionUNetConfig:
    inp_size: Tuple[int, int] = (32, 32)
    inp_chls: int = 3
    time_dim: int = 32
    down_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    up_channels: Tuple[int, ...] = (512, 256, 128, 64, 32)

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp =  nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # x: [B, in_ch, W, H]
        # t: [B, time_dim]
        h = self.bnorm1(self.relu(self.conv1(x))) # [B, out_ch, H, W]
        time_emb = self.relu(self.time_mlp(t)) # [B, out_ch]
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1) # [B, out_ch, 1, 1] for broadcasting
        h = h + time_emb # [B, out_ch, H, W]
        h = self.bnorm2(self.relu(self.conv2(h))) # [B, out_ch, H, W]
        y = self.transform(h) # [B, out_ch, H//2, W//2]
        return y

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp =  nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(2*in_ch, out_ch, kernel_size=3, padding=1) # 2*in_ch for skip connections
        self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # x: [B, in_ch, W, H]
        # t: [B, time_dim]
        h = self.bnorm1(self.relu(self.conv1(x))) # [B, out_ch, H, W]
        time_emb = self.relu(self.time_mlp(t)) # [B, out_ch]
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1) # [B, out_ch, 1, 1] for broadcasting
        h = h + time_emb # [B, out_ch, H, W]
        h = self.bnorm2(self.relu(self.conv2(h))) # [B, out_ch, H, W]
        y = self.transform(h) # upsample: [B, out_ch, H*2, W*2]
        return y

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionUNet(nn.Module):
    """
    A simplified Unet for Diffusion. Not using Attension, GroupNorm etc
    """
    def __init__(self, config: DiffusionUNetConfig):
        super().__init__()
        self.config = config
        down_channels = config.down_channels
        up_channels = config.up_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.time_dim),
            nn.Linear(config.time_dim, config.time_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(config.inp_chls, down_channels[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Downsample(down_channels[i], down_channels[i+1], config.time_dim) for i in range(len(down_channels)-1)
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Upsample(up_channels[i], up_channels[i+1], config.time_dim) for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], config.inp_chls, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1) # add residual as additional channel
            x = up(x, t)
        return self.output(x)

if __name__ == "__main__":
    config = DiffusionUNetConfig()
    model = DiffusionUNet(config)
    print(f"Num params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}Mn")
    model
    x = torch.rand((1, 3, 32, 32))
    t = torch.randint(0, 10, size=(1,))
    y = model(x, t)
    assert y.shape == (1, 3, 32, 32)