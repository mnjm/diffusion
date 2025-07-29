import os
import random
from contextlib import nullcontext
from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm import Diffusion
from model import DiffusionUNet, DiffusionUNetConfig
from utils import (
    EMAModelWrapper,
    get_dataset,
    get_ist_time,
    save_imgs,
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
)

device = torch_get_device()
print(f"using {device}")

# -------------------- Params --------------------------
rng_seed = 31415
dataset = "cifar10" # should be one of 'mnist', 'cifar10', 'tiny-imagenet' or 'landscapes'
clfr_free_guidance = False # classifier free guidance
enable_ema = False
ema_beta = 0.995
noise_steps = 1000
n_epochs = 500
batch_size = 256 if device.type == "cuda" else 32
ema_warmup_steps = 25600 // batch_size
lr = 2e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.01
init_from = 'scratch'
save_every_epoch = 50
vis_every_epoch = 10
vis_n_samples = 10 if clfr_free_guidance else 16
log_dir = "./logs"
enable_wandb = False
wandb_project = f"diffusion-{dataset}"
model_name = wandb_project
run_name = f"{'cfa_' if clfr_free_guidance else ''}{'ema_' if enable_ema else ''}" + get_ist_time()
enable_torch_compile = device.type == "cuda"
enable_tf32 = device.type == "cuda"
torch_amp_dtype = 'bfloat16' if device.type == "cuda" and torch.cuda.is_bf16_supported() else 'float32'
dataloader_workers = 4 if device.type == "cuda" else 0
dataloader_pin_memory = device.type == "cuda"
dataloader_drop_last = True
# ------------------------------------------------------
config = { k:v for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str)) }

torch_amp_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[torch_amp_dtype]

torch_set_seed(rng_seed)
random.seed(rng_seed)

if clfr_free_guidance:
    assert vis_n_samples == 10, "vis_n_samples should be 10 while doing classfier free guidance based training"

model_config = DiffusionUNetConfig()
model_config.n_classes = 10 if clfr_free_guidance else 0
img_size, img_chls, torch_ds = get_dataset(dataset)
if img_size == (24, 24) and img_chls == 1:
    # MNIST requires liter model since its image size is 24, 24 with channels 1
    model_config.in_chls = 1
    model_config.out_chls = 1
    model_config.init_hidden_chls = 64
    model_config.chls_mult_factor = [1, 2, 4]
    model_config.num_res_blocks = 2
    model_config.attention_resolutions = [12]
    model_config.dropout = 0.1
    model_config.time_embed_dim = 256

dataloader = DataLoader(torch_ds, batch_size=batch_size, shuffle=True, drop_last=dataloader_drop_last,
                        pin_memory=dataloader_pin_memory, num_workers=dataloader_workers)
# import matplotlib.pyplot as plt
# imgs, _ = next(iter(dataloader))
# assert imgs.shape[0] > 16
# grid_img = make_grid(imgs[:16, ...], 4, normalize=True).to("cpu").permute(1, 2, 0).numpy()
# plt.imshow(grid_img)
# plt.show()
# import sys; sys.exit(0)

# Model
if init_from == 'scratch':
    model = DiffusionUNet(model_config)
    model.to(device)
else:
    ckpt = torch.load(init_from, map_location=device, weights_only=False)
    assert noise_steps == ckpt['config']['noise_steps'], "Different noise_steps"
    assert dataset == ckpt['config']['dataset'], "Different dataset"
    model = DiffusionUNet(ckpt['model_config'])
    model.to(device)
    state_dict = ckpt['model']
    state_dict = torch_compile_ckpt_fix(state_dict)
    model.load_state_dict(state_dict)
    print(f"Loaded ckpt {init_from}")

print(f"Model num params: {sum(p.numel() for p in model.parameters()):,}")
if enable_torch_compile:
    model = torch.compile(model)

diffusion = Diffusion(noise_steps, img_size=img_size, img_chnls=img_chls, device=device)

optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, fused=device.type=="cuda")

if init_from != 'scratch':
    optimizer.load_state_dict(ckpt['optimizer'])

if enable_tf32:
    torch.set_float32_matmul_precision("high")
amp_ctx = torch.amp.autocast(device_type=device.type, dtype=torch_amp_dtype) if device.type == "cuda" and torch_amp_dtype == torch.bfloat16 else nullcontext()

if enable_wandb:
    load_dotenv()
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project=wandb_project, name=run_name, config=config)
log_dir = Path(log_dir) / model_name / run_name
log_dir.mkdir(parents=True, exist_ok=True)

if enable_ema:
    ema = EMAModelWrapper(model, ema_beta, ema_warmup_steps)
    if init_from != "scratch":
        ema.load(ckpt['ema'])

# training
model.train()
for epoch in range(1, n_epochs + 1):
    t0 = time()
    loss_cum = 0.0
    print(f"Epoch {epoch}/{n_epochs}")

    progress_bar = tqdm(dataloader, dynamic_ncols=True, desc="Training", leave=False)

    for step, batch in enumerate(progress_bar):
        imgs = batch[0].to(device)
        lbls = None
        if clfr_free_guidance and random.random() < 0.9:
            lbls = batch[1].to(device)

        optimizer.zero_grad()

        B, C, H, W = imgs.shape
        t = diffusion.sample_timesteps(B)
        x_noisy, noise = diffusion.forward(imgs, t)

        with amp_ctx:
            noise_pred = model(x_noisy, t, lbls)
            loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()
        loss_cum += loss.item()

        if enable_ema:
            ema.update(model)

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_time = time() - t0
    avg_loss = loss_cum / len(dataloader)
    print(f"Loss: {avg_loss:.4f} Time: {epoch_time:.2f}s")
    if enable_wandb:
        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
            'epoch_time': epoch_time,
        })

    if epoch % vis_every_epoch == 0:
        model.eval()
        with torch.no_grad():
            lbls, cfg_scale = None, 0
            if clfr_free_guidance:
                lbls = torch.arange(0, 10, dtype=torch.long, device=device)
                cfg_scale = 3
            imgs, _ = diffusion.reverse(model, n=vis_n_samples, lbls=lbls, cfg_scale=cfg_scale, amp_ctx=amp_ctx)
            img_path = log_dir / f"{model_name}-{epoch:05d}.png"
            save_imgs(imgs, img_path, nrow=5 if clfr_free_guidance else 0)
            if enable_ema and ema.is_active():
                imgs, _ = diffusion.reverse(ema, n=vis_n_samples, lbls=lbls, cfg_scale=cfg_scale, amp_ctx=amp_ctx)
                img_path = log_dir / f"{model_name}-{epoch:05d}-ema.png"
                save_imgs(imgs, img_path, nrow=5 if clfr_free_guidance else 0)
        print(f"Saved sample images generated to {str(img_path)}")
        model.train()

    if epoch % save_every_epoch == 0:
        ckpt_path = log_dir / f"{model_name}.pt"
        torch.save({
            'model': model.state_dict(),
            'ema': ema.state_dict() if enable_ema else None,
            'model_config': model_config,
            'config': config,
            'epoch': epoch,
            'loss': avg_loss,
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint to {str(ckpt_path)}")