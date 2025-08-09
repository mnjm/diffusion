from contextlib import nullcontext
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
        self.schedule = cfg.schedule

        supported_schedules = { 'linear': self._linear_schedule, 'cosine': self._cosine_schedule }
        assert self.schedule in supported_schedules, "Unknown diffusion beta schedule"
        # pre calculate coeffs
        self.betas = supported_schedules[self.schedule]()
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)
        alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_1_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)
        posterior_var = self.betas * (1. - alphas_bar_prev) / (1. - self.alphas_bar)
        self.sqrt_posterior_var = torch.sqrt(posterior_var)

    def __repr__(self):
        ret = f"Diffusion with {self.timesteps} timesteps {self.schedule} schedule"
        if self.cfg:
            ret += f" with CFG (scale:{self.cfg_scale})"
        return ret

    def _linear_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)

    def _cosine_schedule(self):
        s = 0.008
        T = self.timesteps
        x = torch.linspace(0, T, T+1, device=self.device)
        bar_alphas = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        bar_alphas = bar_alphas / bar_alphas[0]
        betas = 1 - (bar_alphas[1:] / bar_alphas[:-1])
        # return torch.clip(betas, 0.0001, 0.9999)
        return torch.clip(betas, 0, 0.9999)

    @torch.no_grad()
    def forward(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_bar_alpha_t = self.sqrt_alphas_bar[t][:, None, None, None] # (B,1,1,1)
        sqrt_1_minus_bar_alpha_t = self.sqrt_1_minus_alphas_bar[t][:, None, None, None] # (B,1,1,1)
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
        for i in tqdm(reversed(range(1, self.timesteps)), dynamic_ncols=True, desc="Sampling", leave=False, total=self.timesteps-1):
            t = torch.full((n,), i, dtype=torch.long, device=self.device)
            with amp_ctx:
                pred_noise = model(x, t, lbls)
                if self.cfg:
                    uncond_pred_noise = model(x, t, None)
                    pred_noise = torch.lerp(uncond_pred_noise, pred_noise, self.cfg_scale)
            betas_t = self.betas[t][:, None, None, None] # (B,1,1,1)
            sqrt_1_minus_alphas_bar_t = self.sqrt_1_minus_alphas_bar[t][:, None, None, None] # (B,1,1,1)
            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None] # (B,1,1,1)
            x = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_1_minus_alphas_bar_t)
            if i > 1:
                sigma_t = self.sqrt_posterior_var[t][:, None, None, None] # (B,1,1,1)
                x = x + sigma_t * torch.randn_like(x)
            x = x.clamp(-1, 1) # NOTE: This is to retain images in same dist (-1, 1). Otherwise creates maddied images
            if debug and i % debug_stepsize == 0 and step_idx < debug_steps:
                x_debug = ((x.clamp(-1.0, 1) + 1) * 127.5).to("cpu").to(torch.uint8)
                debug_ret[step_idx] = x_debug
                step_idx += 1

        x = ((x + 1) * 127.5).to("cpu").to(torch.uint8)
        return x, debug_ret