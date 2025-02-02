import gradio as gr
from PIL import Image, ImageDraw
import torch
import numpy as np
import cv2
from model import Diffusion
from repaint_loop import repaint_inpaint
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def load_model():
    model = Diffusion(image_size=28, in_channels=1, timesteps=1000, base_dim=64).to(device)

    checkpoint_path = "checkpoints/epoch_99.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        print("Checkpoint loaded successfully.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
        return

    return model


def upsample_image(image_np, scale_factor=10):
    """
    Fictively upsample the image using nearest-neighbor interpolation.
    Supports both grayscale and RGB images.
    """
    height, width, channels = image_np.shape
    upsampled_image = cv2.resize(
        image_np,
        (width * scale_factor, height * scale_factor),
        interpolation=cv2.INTER_NEAREST,
    )

    return upsampled_image


def downsample_mask(mask_np, target_size=(28, 28)):
    """
    Downsample the mask to match the size of the input image (28x28).
    Ensure the mask is a NumPy array before resizing.
    """
    if isinstance(mask_np, dict) and "composite" in mask_np:
        mask_np = mask_np["composite"]

    if mask_np.ndim == 3 and mask_np.shape[2] == 4:  # RGBA mask
        mask_np = mask_np[:, :, 3]

    downsampled_mask = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_NEAREST)
    return downsampled_mask


def inpaint_image(input_image_np, mask_np, model, num_timesteps, U):
    """
    Inpainting function that takes RGB NumPy arrays as input.
    Applies the mask to black out the selected regions.
    """
    folder = "gradio"
    os.makedirs(folder, exist_ok=True)

    mask_np = np.expand_dims(mask_np, axis=-1)
    mask_np = np.repeat(mask_np, 3, axis=-1)  # shapes (H, W, 3)

    input_image_gray = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)
    input_image_gray = np.expand_dims(input_image_gray, axis=-1)

    input_image_tensor = torch.tensor(input_image_gray.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    mask_tensor = torch.tensor(mask_np[:, :, 0]).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0  # shapes (1, 1, H, W) 1 for mask 0 for not mask

    # mask_tensor[:, :, :, :] = 0
    # mask_tensor[:, :, 10:20, 10:20] = 1

    masked_image_tensor = input_image_tensor.clone()
    masked_image_tensor[mask_tensor == 1] = 0

    masked_image__out_np = masked_image_tensor.squeeze(1).permute(1, 2, 0).cpu().numpy()

    masked_image_tensor -= 0.5
    masked_image_tensor /= 0.5

    mask_np = mask_tensor.squeeze().cpu().numpy()
    masked_image_np = masked_image_tensor.squeeze().cpu().numpy()

    plt.imsave(os.path.join(folder, "mask.png"), mask_np, cmap="gray")
    plt.imsave(os.path.join(folder, "masked_image.png"), masked_image_np, cmap="gray")

    print("started doing diffusion")
    inpainted_image_tensor = repaint_inpaint(
        model=model,
        input_image=masked_image_tensor,
        mask=mask_tensor,
        num_timesteps=num_timesteps,
        U=U,
        device=device
    )
    print("done with diffusion")

    plt.imsave(os.path.join(folder, "saved_image.png"), inpainted_image_tensor.squeeze(0).cpu().numpy().squeeze(0), cmap='gray')

    delta = input_image_tensor.squeeze(0).cpu().numpy().squeeze(0)-inpainted_image_tensor.squeeze(0).cpu().numpy().squeeze(0)

    inpainted_image_tensor -= torch.min(inpainted_image_tensor)
    inpainted_image_tensor /= torch.max(inpainted_image_tensor) - torch.min(inpainted_image_tensor)

    plt.imsave(os.path.join(folder, "delta.png"), delta, cmap='gray')

    combined_image = torch.cat((input_image_tensor , mask_tensor, inpainted_image_tensor), dim=3)  # Spoji po širini
    save_image(combined_image, os.path.join(folder, "combined.png"))

    inpainted_image_np = (inpainted_image_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    return masked_image__out_np, inpainted_image_np


folder = "example_images"
os.makedirs(folder, exist_ok=True)
mnist_images = [
    np.array(Image.open(os.path.join(folder, f"img_{i}.png")).convert("RGB"))
    for i in range(10)
]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model()

def demo():
    with gr.Blocks() as demo_interface:
        gr.Markdown("# MNIST Inpainting Demo with Arbitrary Masking")

        with gr.Row():
            # Update the choices to include images from 0 to 9
            image_selector = gr.Radio(
                choices=[f"Image {i}" for i in range(10)],
                label="Select Predefined MNIST Image",
            )
            process_button = gr.Button("Run Inpainting", variant="primary")

        with gr.Row():
            upsampled_image_display = gr.Image(
                label="Selected MNIST Image",
                type="numpy",
                image_mode="RGB",
                interactive=False,
                width=280,
                height=280,
            )

            sketchpad = gr.Sketchpad(label="Draw Mask Here", canvas_size=(280, 280))

            masked_image_display = gr.Image(
                label="Output Image",
                type="numpy",
                image_mode="RGB",
                interactive=False,
                width=280,
                height=280,
            )

        with gr.Row():
            output_image_display = gr.Image(
                label="Output Image",
                type="numpy",
                image_mode="RGB",
                interactive=False,
                width=280,
                height=280,
            )

        def load_selected_image(selected_option):
            selected_index = int(selected_option.split()[-1])
            if 0 <= selected_index < len(mnist_images):
                original_image = mnist_images[selected_index]
            else:
                return None

            return upsample_image(original_image)

        image_selector.change(
            load_selected_image, image_selector, upsampled_image_display
        )

        def process_and_upsample(selected_option, mask_np):
            selected_index = int(selected_option.split()[-1])
            if 0 <= selected_index < len(mnist_images):
                original_image = mnist_images[selected_index]
            else:
                return None

            downsampled_mask = downsample_mask(mask_np)
            masked_image, processed_image = inpaint_image(original_image, downsampled_mask, model, 1000, 5)
            return upsample_image(masked_image), upsample_image(processed_image)

        process_button.click(
            process_and_upsample,
            inputs=[image_selector, sketchpad],
            outputs=(masked_image_display, output_image_display),
        )

    return demo_interface


demo().launch()