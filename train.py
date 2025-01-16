import torch
import torch.nn as nn
from utils import ExponentialMovingAverage
import os


def save_checkpoint(epoch, model, ema, global_steps, checkpoint_dir="checkpoints"):
    """Saves model and EMA states to a checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "global_steps": global_steps,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))


def log_training_step(epoch, step, loss, lr, total_steps, log_freq):
    """Logs training progress at defined intervals."""
    if step % log_freq == 0:
        print(
            f"Epoch [{epoch + 1}], Step [{step}/{total_steps}], Loss: {loss:.5f}, LR: {lr:.6f}"
        )


def train(model, dataloader, optimizer, scheduler, device, epochs, log_freq, ema_decay):
    alpha = 1.0 - ema_decay
    ema = ExponentialMovingAverage(model, alpha=alpha, compute_device=device)
    loss_fn = nn.MSELoss()

    global_steps = 0
    for epoch in range(epochs):
        model.train()
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            noise = torch.randn_like(images)
            t = torch.randint(0, model.timesteps, (images.size(0),), device=device)

            # Forward pass
            x_t = model._forward_diffusion(images, t, noise, training=True)
            pred_noise = model(x_t, t=t, noise=noise, training=True)
            loss = loss_fn(pred_noise, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update EMA
            if global_steps % 10 == 0:
                ema.update_parameters(model)

            global_steps += 1

            # Logging
            log_training_step(
                epoch,
                i,
                loss.item(),
                scheduler.get_last_lr()[0],
                len(dataloader),
                log_freq,
            )

        # Save checkpoint
        save_checkpoint(epoch, model, ema, global_steps)

        # Generate samples
        ema.eval()
        samples = ema.module.sampling(16, device=device)

