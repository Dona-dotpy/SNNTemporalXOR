import torch

## --- Constants / Parameters ---
# General
SEED = 42

# Input
PROB_0, PROB_1 = 0.3, 0.6                           # number of spike to represent 0 and 1
SIGN_A, SIGN_B = 10, 10                             # number of timesteps of signals A and B of XOR
GAP, READOUT = 30, 10                               # numers of timesteps of gap and readout
TOT_TIMESTEPS = SIGN_A + SIGN_B + GAP + READOUT     

# Dataset
BATCH_SIZE = 32

# Neural network
NUM_SAMPLES = 1000
TRAIN_SPLIT = 0.8
INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER = 3, 50, 2
N_ROBUSTNESS_DATASET = 10

LEARNING_RATE = 0.0001
EPOCHS = 1000

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
