import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import cv2


def upsample_image(image_np, scale_factor=10):
    """
    Fictively upsample the image using nearest-neighbor interpolation.
    Supports both grayscale and RGB images.
    """
    height, width, channels = image_np.shape
    upsampled_image = cv2.resize(
        image_np, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_NEAREST)
    return upsampled_image


def downsample_mask(mask_np, target_size=(32, 32)):
    """
    Downsample the mask to match the size of the input image (32x32).
    Ensure the mask is a NumPy array before resizing.
    """
    if isinstance(mask_np, dict) and 'composite' in mask_np:
        mask_np = mask_np['composite']

    if mask_np.ndim == 3 and mask_np.shape[2] == 4:  # RGBA mask
        mask_np = mask_np[:, :, 3]

    downsampled_mask = cv2.resize(
        mask_np, target_size, interpolation=cv2.INTER_NEAREST)
    return downsampled_mask


def inpaint_image(input_image_np, mask_np):
    """
    Inpainting function that takes RGB NumPy arrays as input.
    Applies the mask to black out the selected regions.
    """
    if mask_np.ndim == 2:
        mask_np = np.expand_dims(mask_np, axis=-1)
    mask_np = np.repeat(mask_np, 3, axis=-1)

    inpainted_image = np.where(mask_np == 255, 0, input_image_np)
    return inpainted_image


cifar_images = [
    np.array(Image.open("img_1.png").resize((32, 32)).convert("RGB")),
    np.array(Image.open("img_2.png").resize((32, 32)).convert("RGB")),
    np.array(Image.open("img_4.png").resize((32, 32)).convert("RGB")),
]


def demo():
    with gr.Blocks() as demo_interface:
        gr.Markdown("# CIFAR Inpainting Demo with Sketchpad Masking")

        with gr.Row():
            image_selector = gr.Radio(
                choices=["Image 1", "Image 2", "Image 3"],
                label="Select Predefined CIFAR Image",
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
                original_image = cifar_images[0]
            elif selected_option == "Image 2":
                original_image = cifar_images[1]
            elif selected_option == "Image 3":
                original_image = cifar_images[2]
            else:
                return None

            return upsample_image(original_image)

        image_selector.change(load_selected_image,
                              image_selector, upsampled_image_display)

        def process_and_upsample(selected_option, mask_np):
            if selected_option == "Image 1":
                original_image = cifar_images[0]
            elif selected_option == "Image 2":
                original_image = cifar_images[1]
            elif selected_option == "Image 3":
                original_image = cifar_images[2]
            else:
                return None

            downsampled_mask = downsample_mask(mask_np)
            processed_image = inpaint_image(original_image, downsampled_mask)

            return upsample_image(processed_image)

        process_button.click(
            process_and_upsample,
            inputs=[image_selector, sketchpad],
            outputs=output_image_display
        )

        sketchpad.input(process_and_upsample, inputs=[
                        image_selector, sketchpad], outputs=masked_image_display)

    return demo_interface


demo().launch()
