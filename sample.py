import math
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from ddpm import Diffusion
from model import DiffusionUNet, DiffusionUNetConfig
from utils import (
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
    sample_lbls
)

old_cfg_yml = """
timesteps: 1000
beta_start: 1e-4
beta_end: 0.02
schedule: linear
cfg: # classifier free guidance
    enable: false
    scale: 3.0
"""
rng_seed = 31415

def sample(ckpt_path: Path, n: int = 20, save: bool = False, beta_steps: int = -1, cfg_scale: float = 3.0):
    torch_set_seed(rng_seed)
    device_type = "cuda" if torch.cuda.is_available() else "auto"
    device = torch_get_device(device_type)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'model_config' not in ckpt:
        cfg = ckpt['config']
        model_config = DiffusionUNetConfig(**cfg.model)
        model = DiffusionUNet(model_config)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        print(f"Loaded checkpoint from {ckpt_path}")

        img_size, img_chnls = cfg.dataset.img_size, cfg.dataset.img_chls
        n_classes = cfg.model.n_classes
        diffusion_cfg = cfg.diffusion
        dataset = cfg.dataset.name
    else:
        model_config = ckpt['model_config']
        model = DiffusionUNet(model_config)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        print(f"Loaded checkpoint from {ckpt_path}")
        img_size, img_chnls = (32, 32), 3
        n_classes = 0
        diffusion_cfg = OmegaConf.create(old_cfg_yml)
        dataset = ckpt['config']['dataset']

    diffusion_cfg.cfg.scale = cfg_scale
    if beta_steps != -1:
        diffusion_cfg.timesteps = beta_steps
    diffusion = Diffusion(diffusion_cfg, img_size, img_chnls, device=device)
    print(f"Loaded {diffusion}")
    cfg_enabled = diffusion.cfg
    beta_schedule = diffusion.schedule

    lbls = (
        sample_lbls(n_classes, n, device)
        if n_classes > 0 else
        None
    )
    imgs, _ = diffusion.reverse(model, n, lbls)

    nrow = int(math.ceil(math.sqrt(n))) if n_classes == 0 else n_classes
    grid = make_grid(imgs.to("cpu").float(), nrow=nrow, normalize=True)
    grid = grid.permute(1, 2, 0).numpy()
    plt.figure(facecolor='black')
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    title = f"{dataset} ({beta_schedule=})"
    if cfg_enabled:
        title += " with CFG"
    plt.title(title, color='white')
    if save:
        plt.savefig('sampled.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(description="Sample images using a diffusion model checkpoint.")
    parser.add_argument("ckpt_path", type=Path, help="Path to the model checkpoint file.")
    parser.add_argument("-n", "--num_images", type=int, default=20, help="Number of images to sample. Default is 20.")
    parser.add_argument("-s", "--save", action="store_true", help="Save sampled images. Omit to skip saving.")
    parser.add_argument("-st", "--beta_steps", type=int, default=-1, help="Beta timesteps to use for sampling. default: -1 (use training time beta steps)")
    parser.add_argument("-cs", "--cfg_scale", type=float, default=3.0, help="CFG Scale to use with CFG is enabled")

    args = parser.parse_args()
    sample(args.ckpt_path, args.num_images, args.save, args.beta_steps, args.cfg_scale)