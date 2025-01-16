import torch
from torchvision import datasets, transforms
import os
from torchvision.utils import save_image

def save_output(original_image, masked_image, inpainted_image, index, output_folder):
    """
    Combines original, masked, and inpainted images into a single image and saves it.

    Parameters:
        original_image (torch.Tensor): Original input image.
        masked_image (torch.Tensor): Image with the masked region.
        inpainted_image (torch.Tensor): Result of the inpainting process.
        index (int): Index of the current image in the dataset.
        output_folder (str): Directory where the output images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Combine images horizontally (along the width dimension)
    combined_image = torch.cat((original_image, masked_image, inpainted_image), dim=3)  # Spoji po širini
    
    # Save the combined image
    output_path = os.path.join(output_folder, f"inpainted_combined_{index}.png")
    save_image(combined_image, output_path)

def dataloader(batch_size, image_size=28):
    """Creates a MNIST dataloader. Checks if data already exists locally."""
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ]
    )
    dataset = datasets.MNIST(
        root="./mnist_data",
        train=True,
        download=not os.path.exists("./mnist_data"),
        transform=transform,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
