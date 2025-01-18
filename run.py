import functions
import torch
from model import Diffusion
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import train
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from repaint_loop import repaint_inpaint
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training hyperparameters
    lr = 0.001
    batch_size = 128
    epochs = 10  
    timesteps = 1000
    model_base_dim = 64
    ema_decay = 0.995
    log_freq = 10

    # Load MNIST and initialize model
    dataloader = functions.dataloader(batch_size)
    model = Diffusion(image_size=28, in_channels=1, timesteps=timesteps, base_dim=model_base_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(dataloader))

    # Train the model
    # Uncomment the following line to train the model

    #train.train(model, dataloader, optimizer, scheduler, device, epochs, log_freq, ema_decay)

    
    # Load pretrained model
    checkpoint_path = "checkpoints/epoch_99.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        print("Checkpoint loaded successfully.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
        return
    
    
    mnist_testset = datasets.MNIST(
        root="./mnist_data", 
        train=False, 
        download=True, 
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )

    # Define the output folder for saving results
    output_folder = "repaint_output_v4.1"
    U = 5
    num_timesteps = 80
    
    # Initialize a dictionary to track collected digits
    collected_digits = {digit: False for digit in range(10)}
    collected_count = 0

    # Iterate through the MNIST test dataset
    for index, (input_image, label) in enumerate(mnist_testset):
        # Check if the current digit is already collected
        if not collected_digits[label]: 
            input_image = input_image.unsqueeze(0).to(device)

            # Create a binary mask
            mask = torch.zeros((1, 1, 28, 28), device=device)
            mask[:, :, 10:20, 10:20] = 1    # Define the region to be inpainted

            # Create the masked image
            masked_image = input_image.clone()  
            masked_image[:, :, 10:20, 10:20] = 0  # Fill the masked region with a constant value

            # Perform inpainting using the RePaint loop
            inpainted_image = repaint_inpaint(
                model=model,
                input_image=masked_image,
                mask=mask,
                num_timesteps=num_timesteps,
                U=U,
                device=device
            )

            # Save the combined result (original, masked, inpainted)
            functions.save_output(input_image, masked_image, inpainted_image, label, output_folder)

            # Mark the digit as collected
            collected_digits[label] = True
            collected_count += 1

            # Break the loop if all digits are collected
            if collected_count == 10:
                print("Collected all digits from 0 to 9. Stopping the loop.")
                break


if __name__ == "__main__":
    main()
