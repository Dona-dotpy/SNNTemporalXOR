"""
Global Configuration and Parameter Settings for Temporal XOR SNN.
Serves as the Single Source of Truth for the entire repository, defining:
- Temporal setup: durations of signals A, B, GAP, and terminal Readout.
- Rate coding & noise: spike firing probabilities and noise-aware mixture regimes.
- Network & LIF parameters: layer dimensions and membrane decay factor (beta).
- Optimizer & learning rate scheduler hyperparameter presets.
- Dynamic hardware allocation (CPU, CUDA, MPS) based on batch workload.
"""

## --- Libraries ---
import torch


## --- General ---
SEED = 1542

## --- Input ---
PROB_0, PROB_1 = 0.3, 0.6                           # spike probabilities used to represent 0 and 1
SIGN_A, SIGN_B = 10, 10                             # number of timesteps of signals A and B of XOR
GAP, READOUT = 30, 10                               # numers of timesteps of gap and readout
TOT_TIMESTEPS = SIGN_A + SIGN_B + GAP + READOUT
NOISE_SPIKES = 0     
NOISE_AWARE = False                                 # whether dataset uses noise mixture perturbation

## --- Dataset ---
BATCH_SIZE = 32
NUM_SAMPLES = 1000
TRAIN_SPLIT = 0.8
N_ROBUSTNESS_DATASET = 10

## --- Neurons ---
BETA = 0.979350180923352

## --- Neural network ---

# we put here the best parameters we found from a 62 hours search with the script "optimize_hyperparameters.py"
# Optimal parameters from Test 2 (Nominal 50 neurons, Trial 82)

INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER = 3, 50, 2
LOSS_FN = torch.nn.CrossEntropyLoss
FLUCT_XI = 1.2923776896484005                      # fluctuation-driven initialization scale
OPTIMIZER = torch.optim.AdamW
OPTIMIZER_PARAMS = {
    "lr": 0.004978658660260544,
    "betas": (0.85, 0.953),
    "weight_decay": 1e-06,
    "amsgrad": False
}
SCHEDULER = torch.optim.lr_scheduler.MultiStepLR
SCHEDULER_PARAMS = {
    "milestones": [250, 500, 750],
    "gamma": 0.7664475228425424
}
LEARNING_RATE = 0.004978658660260544              # write in floating point (do not write 1e-3)
EPOCHS = 1000

## --- Device ---
workload = BATCH_SIZE * TOT_TIMESTEPS * HIDDEN_LAYER * max(INPUT_LAYER, OUTPUT_LAYER)   #estabilish a workload to decide if gpu is covenient
if workload > 1000000 and torch.cuda.is_available():
    DEVICE = "cuda"
elif workload > 10000000 and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"