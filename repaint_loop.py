import torch
import torch.nn as nn
import numpy as np
from model import Diffusion
import os

def repaint_inpaint(model, input_image, mask, num_timesteps=1000, U=5, device="cuda"):
    input_image = input_image.to(device)
    mask = mask.to(device)

    # Initialize x_T with Gaussian noise
    xt = torch.randn_like(input_image, device=device)

    # Define beta schedule and precompute values
    betas = np.linspace(0.0001, 0.02, num_timesteps)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod = torch.sqrt(torch.tensor(alphas_cumprod, device=device))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - torch.tensor(alphas_cumprod, device=device))
    betas = torch.tensor(betas, device=device)
    alphas_torch = torch.tensor(alphas, device=device)

    # Main loop
    for t in reversed(range(num_timesteps)):
        with torch.no_grad():  # Avoid retaining computation graphs
            for u in range(U):
                # Generate noise if needed
                epsilon = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

                # Compute x_t-1 (known part)
                xt_known = (
                    sqrt_alphas_cumprod[t] * input_image
                    + sqrt_one_minus_alphas_cumprod[t] * epsilon
                )

                # Compute x_t-1 (unknown part)
                z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
                t_tensor = torch.full((xt.size(0),), t, device=device, dtype=torch.long)
                xt_unknown = (1 / torch.sqrt(alphas_torch[t])) * (
                    xt - betas[t] / sqrt_one_minus_alphas_cumprod[t] * model(xt, t_tensor)
                ) + torch.sqrt(betas[t]) * z

                # Merge known and unknown parts
                xt_merged = mask * xt_unknown + (1 - mask) * xt_known
                xt = xt_merged

                # Diffusion step (if applicable)
                if u < U - 1 and t > 0:
                    xt = torch.sqrt(1 - betas[t - 1]) * xt + torch.sqrt(
                        betas[t - 1]
                    ) * torch.randn_like(xt)

    # Return the inpainted image
    return xt


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Diffusion(image_size=28, in_channels=1, timesteps=1000, base_dim=64).to(device)
#
# checkpoint_path = "checkpoints/epoch_99.pt"
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model"])
#     model.eval()
#     print("Checkpoint loaded successfully.")
# else:
#     print(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
#
# input_image = torch.rand((1, 1, 28, 28))
# mask = torch.rand((1, 1, 28, 28))
# print("started doing diffusion")
# inpainted_image_tensor = repaint_inpaint(
#     model=model,
#     input_image=input_image,
#     mask=mask,
#     num_timesteps=1000,
#     U=5,
#     device=device
# )
# print("done with diffusion")
