import torch

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """
    Implements exponential moving averages for model parameters with a customizable decay factor.
    The formula applied is:
        updated_avg = alpha * existing_avg + (1 - alpha) * current_param
    Utilizes PyTorch's AveragedModel for parameter tracking.
    """

    def __init__(self, model, alpha, compute_device="cpu"):
        def compute_moving_average(existing_avg, current_param, _):
            return alpha * existing_avg + (1 - alpha) * current_param

        super().__init__(model, compute_device, compute_moving_average, use_buffers=True)
