import torch


## --- General ---
SEED = 1542

## --- Input ---
PROB_0, PROB_1 = 0.3, 0.6                           # number of spike to represent 0 and 1
SIGN_A, SIGN_B = 10, 10                             # number of timesteps of signals A and B of XOR
GAP, READOUT = 30, 10                               # numers of timesteps of gap and readout
TOT_TIMESTEPS = SIGN_A + SIGN_B + GAP + READOUT     

## --- Dataset ---
BATCH_SIZE = 32
NUM_SAMPLES = 1000
TRAIN_SPLIT = 0.8
N_ROBUSTNESS_DATASET = 10

## --- Neurons ---
BETA = 0.95

## --- Neural network ---
INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER = 3, 50, 2
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau

LEARNING_RATE = 0.00172                              # write in floating point (do not write 1e-3)
EPOCHS = 1000

## --- Device ---
workload = BATCH_SIZE * TOT_TIMESTEPS * HIDDEN_LAYER * max(INPUT_LAYER, OUTPUT_LAYER)   #estabilish a workload to decide if gpu is covenient
if workload > 1000000 and torch.cuda.is_available():
    DEVICE = "cuda"
elif workload > 10000000 and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

## --- Hyperparameters optimization ---

STUDY_NAME = "learning_rate_optimization"
OPTUNA_DB_PATH = "optimization_results/learning_rate/learning_rate_optimization.db"
TARGET_TRIALS = 3
EPOCHS_OPTIMISATION = 100
CHECKPOINT_INTERVAL = 10

LR_MIN, LR_MAX = 1e-5, 1e-2
HIDDEN_LAYER_MIN, HIDDEN_LAYER_MAX = 20, 100
OPTIMIZERS = ["Adam", "AdamW", "SGD"]   # ADAM (lr, weight_decay, beta1, beta2, amsgrad) - AdamW (lr, weight_decay, beta1, beta2, amsgrad) - SGD (lr, weight_decay, momentum, nesterov)
SCHEDULERS = ["None", "StepLR", "MultiStepLR", "ConstantLR", "LinearLR", "ExponentialLR",
               "CosineAnnealingLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts",
            "PolynomialLR"]
LOSSES = [ "CrossEntropyLoss", "NLLLoss", "MSELoss"]
INITIALIZATIONS = ["default", "xavier_uniform", "xavier_normal",
                    "kaiming_uniform", "kaiming_normal", "orthogonal"]

# Optimizer search space
WEIGHT_DECAY_MIN = 1e-8     # minimum penalization applied to large weights
WEIGHT_DECAY_MAX = 1e-2     # maximum penalization applied to large weights

BETA1_MIN = 0.80            # minimum memory of past gradients
BETA1_MAX = 0.99            # maximum memory of past gradients

BETA2_MIN = 0.95            # minimum memory of past squared gradients
BETA2_MAX = 0.9999          # maximum memory of past squared gradients

MOMENTUM_MIN = 0.0          # minimum contribution of previous SGD updates
MOMENTUM_MAX = 0.99         # maximum contribution of previous SGD updates