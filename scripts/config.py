import torch

## --- Constants / Parameters ---
# General
SEED = 1542

# Input
PROB_0, PROB_1 = 0.3, 0.6                           # number of spike to represent 0 and 1
SIGN_A, SIGN_B = 10, 10                             # number of timesteps of signals A and B of XOR
GAP, READOUT = 30, 10                               # numers of timesteps of gap and readout
TOT_TIMESTEPS = SIGN_A + SIGN_B + GAP + READOUT     

# Dataset
BATCH_SIZE = 32
NUM_SAMPLES = 1000
TRAIN_SPLIT = 0.8
N_ROBUSTNESS_DATASET = 10

# Neurons
BETA = 0.95

# Neural network

INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER = 3, 50, 2
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau


LEARNING_RATE = 0.001                              # write in floating point (do not write 1e-3)
EPOCHS = 1000

# Device
workload = BATCH_SIZE * TOT_TIMESTEPS * HIDDEN_LAYER * max(INPUT_LAYER, OUTPUT_LAYER)   #estabilish a workload to decide if gpu is covenient
if workload > 1000000 and torch.cuda.is_available():
    DEVICE = "cuda"
elif workload > 10000000 and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
