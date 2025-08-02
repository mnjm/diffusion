import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import List

@dataclass
class DiffusionUNetConfig:
    """
    CIFAR 10 Diffusion UNet Config. See Appendix B in DDPM paper
    - Input: 3-channel image of size 32x32
    - Output: same shape as input (3x32x32)
    Config Params
        - init_hidden_chls = 128: base channel width
        - chls_mult_factor = [1, 2, 2, 2]: determines channel dimensions per stage: [128, 256, 512, 1024]
        - num_res_blocks = 2: number of residual blocks per resolution level
        - attention_resolutions = [16]: self-attention added at 16x16 resolution
        - dropout = 0.1: dropout used in residual blocks
        - time_embed_dim = 512: dimension of time-step embedding vector
    """
    name: str
    in_chls: int = 3
    out_chls: int = 3
    init_hidden_chls: int = 128
    chls_mult_factor: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    num_res_blocks: int = 2
    attention_resolutions: List[int] = field(default_factory=lambda: [16])
    attention_head: int = 4
    dropout: float = 0.1
    time_embed_dim: int = 512
    n_classes: int = 0 # > 0 will use classifer free guidance

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings (from Transformer paper).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_chls, out_chls, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_emb_proj = nn.Linear(time_emb_dim, out_chls)

        self.norm1 = nn.GroupNorm(8, in_chls)
        self.conv1 = nn.Conv2d(in_chls, out_chls, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_chls)
        self.conv2 = nn.Conv2d(out_chls, out_chls, 3, padding=1)

        self.dropout = nn.Dropout(dropout)

        if in_chls != out_chls:
            self.shortcut = nn.Conv2d(in_chls, out_chls, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = F.silu(time_emb)
        time_emb = self.time_emb_proj(time_emb)[:, :, None, None]
        h = h + time_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, chls, num_heads=4, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(8, chls)
        self.attn = nn.MultiheadAttention(chls, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Conv2d(chls, chls, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.norm(x).view(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        attn, _ = self.attn(out, out, out)
        out = attn.permute(0, 2, 1).view(b, c, h, w)
        return self.proj(out) + x

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DiffusionUNet(nn.Module):
    def __init__(self, cfg:DiffusionUNetConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.init_hidden_chls
        # Time embedding

        self.time_embed = nn.Sequential(
            nn.Linear(ch, cfg.time_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )

        self.conv_in = nn.Conv2d(cfg.in_chls, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.middle_block = nn.ModuleList()
        inp_chs = [ch]
        ds = 1

        for i, mult in enumerate(cfg.chls_mult_factor):
            out_ch = mult * cfg.init_hidden_chls
            for _ in range(cfg.num_res_blocks):
                block = [ResidualBlock(ch, out_ch, cfg.time_embed_dim, cfg.dropout)]
                if ds in cfg.attention_resolutions:
                    block.append(AttentionBlock(out_ch))
                self.down_blocks.append(nn.ModuleList(block))
                inp_chs.append(out_ch)
                ch = out_ch
            if i != len(cfg.chls_mult_factor) - 1: # Dont downsample last layer
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                inp_chs.append(ch)
                ds *= 2

        # Middle block
        self.middle_block.extend([
            ResidualBlock(ch, ch, cfg.time_embed_dim, cfg.dropout),
            AttentionBlock(ch, cfg.attention_head),
            ResidualBlock(ch, ch, cfg.time_embed_dim, cfg.dropout)
        ])

        # Upsampling layers
        for i, mult in reversed(list(enumerate(cfg.chls_mult_factor))):
            out_ch = mult * cfg.init_hidden_chls
            for j in range(cfg.num_res_blocks + 1):
                skip_ch = inp_chs.pop()
                block = [ResidualBlock(ch + skip_ch, out_ch, cfg.time_embed_dim, cfg.dropout)]
                if ds in cfg.attention_resolutions:
                    block.append(AttentionBlock(out_ch))
                if i and j == cfg.num_res_blocks:
                    block.append(Upsample(out_ch))
                    ds //= 2
                self.up_blocks.append(nn.ModuleList(block))
                ch = out_ch

        # Final layers
        self.out_norm = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, cfg.out_chls, 3, padding=1)

        # Classifier Free Guidance
        if cfg.n_classes > 0:
            self.lbl_emb = nn.Embedding(cfg.n_classes, cfg.time_embed_dim)

    def forward(self, x, t, y=None):
        t_emb = self.time_embed(get_timestep_embedding(t, self.cfg.init_hidden_chls))
        if y is not None: # Classifier Free Guidance
            t_emb += self.lbl_emb(y)
        h = self.conv_in(x)
        skips = [h]

        for blocks in self.down_blocks:
            for layer in blocks:
                h = layer(h, t_emb) if isinstance(layer, ResidualBlock) else layer(h)
            skips.append(h)

        for layer in self.middle_block:
            h = layer(h, t_emb) if isinstance(layer, ResidualBlock) else layer(h)

        for blocks in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in blocks:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)

        return self.conv_out(F.silu(self.out_norm(h)))

if __name__ == "__main__":
    config = DiffusionUNetConfig()
    model = DiffusionUNet(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params // 1e6:.2f}M")
    print("Target: ~35.7M parameters")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))

    with torch.no_grad():
        output = model(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")