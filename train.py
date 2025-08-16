import os
import random
import logging
from contextlib import nullcontext
from pathlib import Path
from time import time
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from ddpm import Diffusion
from model import DiffusionUNet, DiffusionUNetConfig
from utils import (
    EMAModelWrapper,
    get_dataset,
    save_imgs,
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
    get_ist_time_now,
    sample_lbls
)
OmegaConf.register_new_resolver("now_ist", get_ist_time_now)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(config):
    logger = logging.getLogger(__name__)
    device = torch_get_device(config.device_type)
    logger.info(f"Using {device}")
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    run_name = hydra.core.hydra_config.HydraConfig.get().job.name
    autocast_dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[config.autocast_dtype]

    torch_set_seed(config.rng_seed)
    random.seed(config.rng_seed)

    if not config.diffusion.cfg.enable:
        config.model.n_classes = 0
    model_config = DiffusionUNetConfig(**config.model)

    torch_ds = get_dataset(config.dataset)
    logger.info(f"Loading {config.dataset.name} dataset")
    dataloader = DataLoader(
        torch_ds, batch_size=config.batch_size, shuffle=True, drop_last=config.dataloader.drop_last,
        pin_memory=config.dataloader.pin_memory, num_workers=config.dataloader.workers
    )
    # import matplotlib.pyplot as plt
    # imgs, lbls = next(iter(dataloader))
    # assert imgs.shape[0] > 16
    # grid_img = make_grid(imgs[:16, ...], 4, normalize=True).to("cpu").permute(1, 2, 0).numpy()
    # plt.title(f",".join(str(x) for x in lbls.numpy().reshape(-1)[:16]))
    # plt.imshow(grid_img)
    # plt.show()
    # import sys; sys.exit(0)

    start_epoch = 1
    if config.init_from == 'scratch':
        model = DiffusionUNet(model_config)
        model.to(device)
    else:
        ckpt = torch.load(config.init_from, map_location=device, weights_only=False)
        ckpt_cfg = ckpt['config']
        model_config = DiffusionUNetConfig(**ckpt_cfg.model)
        assert config.diffusion.timesteps == ckpt_cfg.diffusion.timesteps, f"Different timesteps: ckpt timesteps {ckpt_cfg.diffusion.timesteps}"
        assert model_config.n_classes > 0 if config.diffusion.cfg.enable else model_config.n_classes == 0, f"Incompatible model {config.diffusion.cfg.enable=} {model_config.n_classes=}"
        assert config.dataset.name == ckpt_cfg.dataset.name, f"Different dataset: {ckpt_cfg.dataset.name}"
        model = DiffusionUNet(model_config)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        logger.info(f"Loaded checkpoint from {config.init_from}")
        start_epoch = ckpt['epoch']

    logger.info(f"Model type: {model_config.name} params: {sum(p.numel() for p in model.parameters()):,}")
    if config.torch_compile:
        if config.diffusion.cfg.enable:
            torch._dynamo.config.force_parameter_static_shapes = False
        model = torch.compile(model)

    diffusion = Diffusion(
        config.diffusion, img_size=config.dataset.img_size,
        img_chnls=config.dataset.img_chls, device=device
    )
    logger.info(f"Initializing {diffusion}")
    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    if config.init_from != 'scratch':
        optimizer.load_state_dict(ckpt['optimizer'])

    if config.enable_tf32:
        torch.set_float32_matmul_precision("high")
    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=autocast_dtype)
        if device.type == "cuda" and autocast_dtype == torch.bfloat16
        else nullcontext()
    )

    if config.logging.wandb.enable:
        load_dotenv()
        wnb_key = os.getenv('WANDB_API_KEY')
        assert wnb_key is not None, "WANDB_API_KEY not loaded in env"
        wandb.login(key=wnb_key)
        wandb.init(
          project=config.logging.wandb.project,
          name=run_name,
          config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )
        wandb.define_metric("epoch")
        wandb.define_metric("loss", step_metric="epoch")
        wandb.define_metric("epoch_time", step_metric="epoch")

    if config.ema.enable:
        ema = EMAModelWrapper(model, config.ema.beta, config.ema.warmup_samples // config.batch_size)
        if config.init_from != "scratch":
            ema.load(ckpt['ema'])

    model.train()
    p_keep= 1. - config.diffusion.cfg.p_drop # prob at which labels are retained during cfg enabled training
    for epoch in range(start_epoch, config.n_epochs + 1):
        t0 = time()
        loss_cum = 0.0
        logger.info(f"Epoch {epoch}/{config.n_epochs}")

        progress_bar = tqdm(dataloader, dynamic_ncols=True, desc="Training", leave=False)

        for step, batch in enumerate(progress_bar):
            imgs = batch[0].to(device)
            lbls = None
            if config.diffusion.cfg.enable and random.random() < p_keep:
                lbls = batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)

            B, C, H, W = imgs.shape
            t = diffusion.sample_timesteps(B)
            x_noisy, noise = diffusion.forward(imgs, t)

            with amp_ctx:
                noise_pred = model(x_noisy, t, lbls)
                loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()
            loss_cum += loss.item()

            if config.ema.enable:
                ema.update(model)

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time() - t0
        avg_loss = loss_cum / len(dataloader)
        logger.info(f"Loss: {avg_loss:.4f} Time: {epoch_time:.2f}s")
        if config.logging.wandb.enable:
            wandb.log({
                'epoch': epoch,
                'loss': avg_loss,
                'epoch_time': epoch_time,
            })

        last_epoch = epoch == config.n_epochs
        if last_epoch or epoch % config.vis_every_epoch == 0:
            model.eval()
            with torch.no_grad():
                lbls = None
                if config.diffusion.cfg.enable:
                    lbls = sample_lbls(config.dataset.n_classes, config.vis_n_samples, device)
                imgs, _ = diffusion.reverse(model, n=config.vis_n_samples, lbls=lbls, amp_ctx=amp_ctx)
                img_path = log_dir / f"{config.model_name}-{epoch:05d}.png"
                img = save_imgs(imgs, img_path).permute(1, 2, 0).numpy()
                if config.logging.wandb.enable and config.logging.wandb.log_imgs:
                    wandb.log({
                        "samples": wandb.Image(img),
                    }, step=epoch)
                if config.ema.enable and ema.is_active():
                    imgs, _ = diffusion.reverse(ema, n=config.vis_n_samples, lbls=lbls, amp_ctx=amp_ctx)
                    img_path = log_dir / f"{config.model_name}-{epoch:05d}-ema.png"
                    img = save_imgs(imgs, img_path).permute(1, 2, 0).numpy()
                    if config.logging.wandb.enable and config.logging.wandb.log_imgs:
                        wandb.log({
                            "ema_samples": wandb.Image(img),
                        }, step=epoch)
            logger.info(f"Saved sample images generated to {str(img_path)}")
            model.train()

        if last_epoch or epoch % config.save_every_epoch == 0:
            ckpt_path = log_dir / f"{config.model_name}.pt"
            torch.save({
                'model': model.state_dict(),
                'ema': ema.state_dict() if config.ema.enable else None,
                'config': config,
                'epoch': epoch,
                'loss': avg_loss,
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {str(ckpt_path)}")

if __name__ == "__main__":
    main()