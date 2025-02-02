import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm


class Diffusion(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=64,
        dim_mults=[2, 4],
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.model = Unet(
            timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults
        )

    def forward(self, x, t=None, noise=None, training=False):
        if training==True:
            t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
            x_t = self._forward_diffusion(x, t, noise, training==True)
            pred_noise = self.model(x_t, t)
            return pred_noise
        elif training==False:
            noise = torch.randn_like(x) 
            x_t = self._forward_diffusion(x, t, noise)
            pred_noise = self.model(x_t, t)
            return pred_noise

   
    @torch.no_grad()
    def sampling(self, n_samples, device="cuda"):
        x_t = torch.randn(
            (n_samples, self.in_channels, self.image_size, self.image_size)
        ).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            x_t = self._reverse_diffusion(x_t, t, noise)

        x_t = (x_t + 1.0) / 2.0  # [-1,1] to [0,1]
        return x_t

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise, training=False):
        assert x_0.shape == noise.shape
        if training==True:
            return self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0 + \
                self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * noise
        elif training==False: 
            return (
                self.sqrt_alphas_cumprod.gather(0, t).reshape(x_0.size(0), 1, 1, 1) * x_0
                + self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(
                    x_0.size(0), 1, 1, 1
                )
                * noise
            )

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        return mean + std * noise
