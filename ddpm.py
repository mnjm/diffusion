from contextlib import nullcontext
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Diffusion:

    def __init__(self, cfg, img_size=(32, 32), img_chnls=3, device="cpu"):
        self.timesteps = cfg.timesteps
        self.beta_start = cfg.beta_start
        self.beta_end = cfg.beta_end
        self.cfg = cfg.cfg.enable
        self.cfg_scale = cfg.cfg.scale
        self.img_size = img_size
        self.img_chnls = img_chnls
        self.device = device

        supported_schedules = [ "linear", "cosine"]
        assert cfg.schedule in supported_schedules, "Unknown diffusion beta schedule"
        if cfg.schedule == "linear":
            self._linear_schedule()
        else:
            self._cosine_schedule()

        logger.info(f"Initializing Diffusion with {self.timesteps} noise steps")
        logger.info(f"CFG enabled: {self.cfg}, scale: {self.cfg_scale}")

    def _linear_schedule(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.bar_alphas = torch.cumprod(self.alphas, dim=0)

    def _cosine_schedule(self):
        s = 0.008
        T = self.timesteps
        t = torch.arange(0, T + 1, dtype=torch.float32, device=self.device)
        f_t = torch.cos(((t / T) + s) / (1 + s) * torch.pi / 2) ** 2
        self.bar_alphas = f_t / f_t[0]
        self.betas = 1 - (self.bar_alphas[1:] / self.bar_alphas[:-1])
        # Clip to prevent singularities (as mentioned in the paper)
        self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        self.alphas = 1.0 - self.betas

    def forward(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_bar_alpha_t = torch.sqrt(self.bar_alphas[t])[:, None, None, None] # (B,1,1,1)
        sqrt_1_minus_bar_alpha_t = torch.sqrt(1.0 - self.bar_alphas[t])[:, None, None, None] # (B,1,1,1)
        return sqrt_bar_alpha_t * x_0 + sqrt_1_minus_bar_alpha_t * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=self.device)

    @torch.no_grad()
    def reverse(self, model, n=1, lbls=None, debug=False, debug_steps=10, amp_ctx=None):
        assert lbls is not None if self.cfg else lbls is None, f"CFG is ${self.cfg} but labels is {lbls}"
        amp_ctx = nullcontext() if amp_ctx is None else amp_ctx
        x = torch.randn((n, self.img_chnls, *self.img_size), device=self.device)

        debug_steps = min(debug_steps, self.timesteps)
        debug_stepsize = max(1, self.timesteps // debug_steps)

        debug_ret = None
        if debug:
            debug_ret = torch.zeros((debug_steps, n, self.img_chnls, *self.img_size), dtype=torch.uint8, device="cpu")

        step_idx = 0
        rev_t = list(reversed(range(1, self.timesteps)))
        for i in tqdm(rev_t, dynamic_ncols=True, desc="Sampling", leave=False):
            t = torch.full((n,), i, dtype=torch.long, device=self.device)
            with amp_ctx:
                pred_noise = model(x, t, lbls)
                if self.cfg:
                    uncond_pred_noise = model(x, t, None)
                    pred_noise = torch.lerp(uncond_pred_noise, pred_noise, self.cfg_scale)
            alpha_t = self.alphas[t][:, None, None, None] # (B,1,1,1)
            bar_alpha_t = self.bar_alphas[t][:, None, None, None] # (B,1,1,1)
            beta_t = self.betas[t][:, None, None, None] # (B,1,1,1)
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / (torch.sqrt(1 - bar_alpha_t))) * pred_noise) + torch.sqrt(beta_t) * noise

            if debug and i % debug_stepsize == 0 and step_idx < debug_steps:
                x_debug = (((x.clamp(-1.0, 1.0) + 1) * 0.5) * 255).to("cpu").to(torch.uint8)
                debug_ret[step_idx] = x_debug
                step_idx += 1

        x = x.clamp(-1.0, 1.0)
        x = (((x + 1) * 0.5) * 255).to("cpu").to(torch.uint8)
        return x, debug_ret