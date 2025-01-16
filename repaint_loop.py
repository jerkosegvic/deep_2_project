import torch
import torch.nn as nn
import numpy as np


def repaint_inpaint(model, input_image, mask, num_timesteps=1000, U=5, device="cuda"):
    input_image = input_image.to(device)
    mask = mask.to(device)

    # Initialize x_T with Gaussian noise
    xt = torch.randn_like(input_image, device=device)

    # Define beta schedule
    betas = np.linspace(0.0001, 0.02, num_timesteps)

    # Precompute alpha and beta values
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.pad(
        alphas_cumprod[:-1], (1, 0), "constant", constant_values=1
    )
    sqrt_alphas_cumprod = torch.sqrt(torch.tensor(alphas_cumprod, device=device))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(
        1 - torch.tensor(alphas_cumprod, device=device)
    )
    betas = torch.tensor(betas, device=device)

    for t in reversed(range(num_timesteps)):
        # Inner loop
        for u in range(U):
            # Generate noise if needed
            epsilon = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

            # Compute x_t-1(known)
            xt_known = (
                sqrt_alphas_cumprod[t] * input_image
                + sqrt_one_minus_alphas_cumprod[t] * epsilon
            )

            # Compute x_t-1(unknown)
            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

            # Add batch dimension to `t`
            t_tensor = torch.full((xt.size(0),), t, device=device, dtype=torch.long)

            xt_unknown = (1 / torch.sqrt(torch.tensor(alphas[t], device=device))) * (
                xt - betas[t] / sqrt_one_minus_alphas_cumprod[t] * model(xt, t_tensor)
            ) + torch.sqrt(betas[t]) * z

            # Merge known and unknown parts
            xt_merged = mask * xt_unknown + (1 - mask) * xt_known

            # Overwrite xt for the next step
            xt = xt_merged.clone()

            # If u < U and t > 1, perform one diffusion step
            if u < U - 1 and t > 0:
                xt = torch.sqrt(1 - betas[t - 1]) * xt + torch.sqrt(
                    betas[t - 1]
                ) * torch.randn_like(xt)

    # Return the inpainted image
    return xt
