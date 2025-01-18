import gradio as gr
from PIL import Image, ImageDraw
import torch
import numpy as np
import cv2
from model import Diffusion
from repaint_loop import repaint_inpaint
import os


def load_model():
    model = Diffusion(image_size=28, in_channels=1,
                      timesteps=1000, base_dim=64).to(device)

    checkpoint_path = "checkpoints/epoch_99.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        print("Checkpoint loaded successfully.")
    else:
        print(
            f"Checkpoint not found at {checkpoint_path}. Please run training first.")
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
    Downsample the mask to match the size of the input image (32x32).
    Ensure the mask is a NumPy array before resizing.
    """
    if isinstance(mask_np, dict) and "composite" in mask_np:
        mask_np = mask_np["composite"]

    if mask_np.ndim == 3 and mask_np.shape[2] == 4:  # RGBA mask
        mask_np = mask_np[:, :, 3]

    downsampled_mask = cv2.resize(
        mask_np, target_size, interpolation=cv2.INTER_NEAREST)
    return downsampled_mask


def inpaint_image(input_image_np, mask_np, model, num_timesteps, U):
    """
    Inpainting function that takes RGB NumPy arrays as input.
    Applies the mask to black out the selected regions.
    """

    mask_np = np.expand_dims(mask_np, axis=-1)
    mask_np = np.repeat(mask_np, 3, axis=-1)  # shapes (H, W, 3)

    input_image_gray = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)
    input_image_gray = np.expand_dims(input_image_gray, axis=-1)

    input_image_tensor = torch.tensor(input_image_gray.transpose(
        2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    mask_tensor = torch.tensor(mask_np[:, :, 0]).unsqueeze(0).unsqueeze(
        0).float().to(device) / 255.0  # shapes (1, 1, H, W)

    masked_image_tensor = input_image_tensor.clone()
    masked_image_tensor[mask_tensor == 1] = 0

    # print(masked_image_tensor.shape, mask_tensor.shape)

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
    # convert the output back to a NumPy array
    inpainted_image_np = (inpainted_image_tensor.squeeze(
        0).cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    print("Output shape:")
    print(inpainted_image_np.shape)

    return inpainted_image_np


def mask_image(input_image_np, mask_np):
    """
    Draw mask over image
    """
    if mask_np.ndim == 2:
        mask_np = np.expand_dims(mask_np, axis=-1)
    mask_np = np.repeat(mask_np, 3, axis=-1)

    inpainted_image = np.where(mask_np == 255, 0, input_image_np)
    return inpainted_image


mnist_images = [
    np.array(Image.open("img_1.png").resize((28, 28)).convert("RGB")),
    np.array(Image.open("img_2.png").resize((28, 28)).convert("RGB")),
    np.array(Image.open("img_3.png").resize((28, 28)).convert("RGB")),
]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model()


def demo():
    with gr.Blocks() as demo_interface:
        gr.Markdown("# MNIST Inpainting Demo with Arbitrary Masking")

        with gr.Row():
            image_selector = gr.Radio(
                choices=["Image 1", "Image 2", "Image 3"],
                label="Select Predefined MNIST Image",
            )
            process_button = gr.Button("Run Inpainting", variant="primary")

        with gr.Row():
            with gr.Column():
                upsampled_image_display = gr.Image(
                    label="Selected CIFAR Image",
                    type="numpy",
                    image_mode="RGB",
                    interactive=False,
                    width=280,
                    height=280,
                )

                sketchpad = gr.Sketchpad(
                    label="Draw Mask Here",
                    canvas_size=(280, 280)
                )

            masked_image_display = gr.Image(
                label="Masked Image",
                type="numpy",
                image_mode="RGB",
                interactive=False,
                width=280,
                height=280,
            )

            output_image_display = gr.Image(
                label="Output Image",
                type="numpy",
                image_mode="RGB",
                interactive=False,
                width=280,
                height=280,
            )

        def load_selected_image(selected_option):
            if selected_option == "Image 1":
                original_image = mnist_images[0]
            elif selected_option == "Image 2":
                original_image = mnist_images[1]
            elif selected_option == "Image 3":
                original_image = mnist_images[2]
            else:
                return None

            return upsample_image(original_image)

        image_selector.change(
            load_selected_image, image_selector, upsampled_image_display
        )

        def process_and_upsample(selected_option, mask_np):
            if selected_option == "Image 1":
                original_image = mnist_images[0]
            elif selected_option == "Image 2":
                original_image = mnist_images[1]
            elif selected_option == "Image 3":
                original_image = mnist_images[2]
            else:
                return None

            downsampled_mask = downsample_mask(mask_np)
            processed_image = inpaint_image(
                original_image, downsampled_mask, model, 1000, 5)
            return upsample_image(processed_image)

        def add_mask(selected_option, mask_np):
            if selected_option == "Image 1":
                original_image = mnist_images[0]
            elif selected_option == "Image 2":
                original_image = mnist_images[1]
            elif selected_option == "Image 3":
                original_image = mnist_images[2]
            else:
                return None

            downsampled_mask = downsample_mask(mask_np)
            processed_image = mask_image(original_image, downsampled_mask)

            return upsample_image(processed_image)

        process_button.click(
            process_and_upsample,
            inputs=[image_selector, sketchpad],
            outputs=output_image_display,
        )

        sketchpad.input(add_mask, inputs=[
            image_selector, sketchpad], outputs=masked_image_display)

    return demo_interface


demo().launch()
