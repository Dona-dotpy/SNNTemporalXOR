"""
Models and classes for the Temporal XOR project.

This module contains the object-oriented components required for the 
Spiking Neural Network (SNN) pipeline. It defines the network architectures, 
custom dataset handlers, and any custom layers used to solve the Temporal XOR task.

Classes:
    * XORSNN: Main Spiking Neural Network architecture with LIF neurons.
    * TemporalXORDataset: Custom PyTorch Dataset for generating and loading XOR sequences.
"""

import functions, config

import torch
import torch.nn as nn
import snntorch as snn

from torch.utils.data import Dataset
from snntorch import utils

class TemporalXORDataset(Dataset):
    """Generates the complete dataset for the temporal XOR task.

    This class constructs a static dataset, meaning all samples and labels are 
    generated once during initialization and remain fixed in memory. 

    Note:
        It is possible to make this dataset dynamic (generating new samples 
        on-the-fly) by moving the data generation logic inside the `__getitem__` 
        method. However, this approach is generally not recommended as it prevents 
        consistent training epochs and validation splits.
    """
    def __init__(self, num_samples=config.NUM_SAMPLES):
        self.num_samples = num_samples
        self.samples = []
        self.labels = []

        # Let's fill the dataset with the number of samples we estabilished
        for _ in range(num_samples):
            A, B = torch.randint(0,2, (1,)).item(), torch.randint(0,2, (1,)).item()
            Data, label = functions.XOR_no_noise(A, B)
            self.samples.append(Data)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

class XORSNN(nn.Module):
    """
    Spiking Neural Network (SNN) with a single hidden layer, based on LIF 
    (Leaky Integrate-and-Fire) neurons. Suitable for time-series processing.

    Architecture:
        1. Linear Layer: (input_layer -> hidden_layer)
        2. Leaky Layer (Hidden): LIF neurons with decay beta=0.9.
        3. Linear Layer: (hidden_layer -> output_layer)
        4. Leaky Layer (Output): LIF neurons with decay beta=0.9.

    Temporal Behavior (init_hidden=True):
        The Leaky layers internally store the membrane potential. The forward method 
        is designed to receive the input of a **single time step (t)**. 
        To process a full sequence, the model must be called inside a temporal 
        loop (e.g., `for t in range(time_steps):`).

    Args:
        input_layer (int): Number of input features (e.g., 3 for Inputs A, B + Control).
        hidden_layer (int): Number of neurons in the hidden layer (e.g., 50).
        output_layer (int): Number of output neurons (e.g., 2).

    Input `x` in the forward pass:
        x (torch.Tensor): Tensor of shape `[batch_size, input_features, time_steps]`.
                          Represents the input data at a specific, single time step.
                          (Example: `[32, 3, 60]` if the batch size is 32 and inputs are 3 and 
                          there are a total 60 time steps).

    Returns:
        tuple (spk, mem): Since `output=True` is set in the final layer, it returns:
            - spk (torch.Tensor): The output spikes (0.0 or 1.0). Shape: `[batch_size, output_layer]`.
            - mem (torch.Tensor): The updated membrane potential. Shape: `[batch_size, output_layer]`.
    """
    def __init__(self, input_layer=config.INPUT_LAYER, hidden_layer=config.HIDDEN_LAYER, output_layer=config.OUTPUT_LAYER):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_layer, hidden_layer),
            snn.Leaky(beta=0.9, init_hidden = True),                           # init_hidden = True stores the mebrane potential between different calls
            nn.Linear(hidden_layer, output_layer),
            snn.Leaky(beta=0.9, init_hidden = True, output = True)             # output = True gives back the membrane potential, too
        )

    def forward(self, x):

        # Check if the tensor dimensionality is correct, otherwise raise a ValueError
        if x.shape[1] != config.INPUT_LAYER:
            raise ValueError(f"No matching between the input layer size required ({config.INPUT_LAYER}) and yours ({x.shape[1]})")

        # Reset of LIF membrane in order to avoid influence of previous batch
        utils.reset(self.net)

        # forward loop
        spk_out = []
        mem_out = []
        for t in range(x.shape[2]):
            spk, mem = self.net(x[:, :, t])
            spk_out.append(spk)
            mem_out.append(mem)

        # return the list converted into a tensor with the same input dimension
        return torch.stack(spk_out, dim=2), torch.stack(mem_out, dim=2)