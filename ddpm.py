import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_torch_device, get_index_from_list

device = get_torch_device()

class Diffusion:
    def __init__(self, T, beta_start=1e-4, beta_end=0.02, img_size=(32, 32), img_chnls=3):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_chnls = img_chnls
        
        self.betas = self.prep_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.bar_alphas = torch.cumprod(self.alphas, dim=0)
        self.sqrt_bar_alphas = torch.sqrt(self.bar_alphas)
        self.sqrt_1_minus_bar_alphas = torch.sqrt(1 - self.bar_alphas)
        self.bar_alphas_prev = F.pad(self.bar_alphas[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_bar_alphas = torch.sqrt(self.bar_alphas)
        posterior_variance = self.betas * (1. - self.bar_alphas_prev) / (1. - self.bar_alphas)
        self.sqrt_posterior_variance = torch.sqrt(posterior_variance)
    
    def prep_beta_schedule(self):
        # TODO: Implement other schedulers, cosine, quadratic, sigmoidal
        return torch.linspace(self.beta_start, self.beta_end, self.T)
    
    def forward(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_bar_alpha_t = get_index_from_list(self.sqrt_bar_alphas, t, x_0.shape)
        sqrt_1_minus_bar_alpha_t = get_index_from_list(self.sqrt_1_minus_bar_alphas, t, x_0.shape)
        return sqrt_bar_alpha_t * x_0 + sqrt_1_minus_bar_alpha_t * noise, noise
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.T, size=(n,))
    
    @torch.no_grad()
    def reverse_single_step(self, x, model, t):
        beta_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_1_minus_bar_alpha_t = get_index_from_list(self.sqrt_1_minus_bar_alphas, t, x.shape)
        sqrt_recip_alpha_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alpha_t * (x - beta_t * model(x, t) / sqrt_1_minus_bar_alpha_t)
        sqrt_posterior_variance_t = get_index_from_list(self.sqrt_posterior_variance, t, x.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + sqrt_posterior_variance_t * noise
    
    @torch.no_grad()
    def reverse(self, model, n, debug=False, debug_steps=10):
        # sample noise
        x = torch.randn((n, self.img_chnls, *self.img_size), device=device)
        debug_stepsize = self.T // debug_steps
        debug_ret = torch.zeros((debug_stepsize, n, self.img_chnls, *self.img_size), dtype=torch.uint8, device="cpu")
        for i in tqdm(range(0, self.T)[:, :, -1], ncols=100):
            t = (torch.ones(n) * i).long().to(device)
            x = self.reverse_single_step(x, model, t)
            x = x.clamp(x, -1.0, 1.0)
            if debug and i % debug_stepsize == 0:
                x_ = (((x + 1) * 0.5) * 255).to("cpu").type(torch.uint8)
                debug_ret[i//debug_stepsize] = x_
        return x, debug_ret