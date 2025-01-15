import torch
import torch.nn as nn
import numpy as np

def repaint_inpaint(model, x0, mask, num_timesteps=1000, U=5):
    """
    Inpainting using the RePaint approach.
    
    Parameters:
    - model: Pretrained diffusion model
    - x0: Original image with masked areas
    - mask: Binary mask indicating missing areas (1 for missing, 0 for known)
    - num_timesteps: Number of diffusion steps
    - U: Number of inner loop iterations
    
    Returns:
    - x0: Inpainted image
    """
    # Initialize x_T with Gaussian noise
    xt = torch.randn_like(x0)
    
    # Precompute alpha and beta values
    alphas = np.linspace(0.0001, 0.02, num_timesteps)  # Example schedule
    alphas_cumprod = np.cumprod(1 - alphas)
    alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), 'constant', constant_values=1)
    sqrt_alphas_cumprod = torch.sqrt(torch.tensor(alphas_cumprod, device=x0.device))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - torch.tensor(alphas_cumprod, device=x0.device))
    betas = 1 - alphas_cumprod / alphas_cumprod_prev
    
    for t in reversed(range(num_timesteps)):
        # Inner loop
        for u in range(U):
            # Generate noise if needed
            epsilon = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            
            # Compute x_t-1(known)
            xt_known = sqrt_alphas_cumprod[t] * x0 + sqrt_one_minus_alphas_cumprod[t] * epsilon
            
            # Compute x_t-1(unknown)
            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            xt_unknown = (1 / torch.sqrt(torch.tensor(alphas[t], device=x0.device))) * (
                xt - betas[t] / sqrt_one_minus_alphas_cumprod[t] * model(xt, t)
            ) + torch.sqrt(torch.tensor(betas[t], device=x0.device)) * z
            
            # Merge known and unknown parts
            xt_merged = mask * xt_unknown + (1 - mask) * xt_known
            
            # Overwrite xt for the next step
            xt = xt_merged.clone()
            
            # If u < U and t > 1, perform one diffusion step
            if u < U - 1 and t > 0:
                xt = torch.sqrt(1 - betas[t - 1]) * xt + torch.sqrt(betas[t - 1]) * torch.randn_like(xt)
    
    return xt
