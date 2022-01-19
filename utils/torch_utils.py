import numpy as np
import torch


def To_device(device):
    def to_device(input, device=device):
        if isinstance(input, (tuple, list)):
            return [to_device(x) for x in input]
        if type(input) == np.ndarray:
            input = torch.tensor(input)
        return input.to(device, non_blocking=True)

    return to_device
