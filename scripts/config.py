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

# Scheduler search space
STEP_SIZE_MIN_FRAC = 0.05              # minimum fraction of epochs between two lr reductions
STEP_SIZE_MAX_FRAC = 0.50              # maximum fraction of epochs between two lr reductions
STEP_GAMMA_MIN = 0.1                   # strongest StepLR reduction factor
STEP_GAMMA_MAX = 0.9                   # weakest StepLR reduction factor

MULTISTEP_PATTERNS = [                  # possible epoch fractions where lr is reduced
    "30_60",
    "50_75",
    "25_50_75",
    "40_70_90"
]
MULTISTEP_GAMMA_MIN = 0.1              # strongest MultiStepLR reduction factor
MULTISTEP_GAMMA_MAX = 0.9              # weakest MultiStepLR reduction factor

CONSTANT_FACTOR_MIN = 0.05             # minimum initial lr scaling factor
CONSTANT_FACTOR_MAX = 0.9              # maximum initial lr scaling factor
CONSTANT_ITERS_MIN_FRAC = 0.01         # minimum fraction of epochs with scaled lr
CONSTANT_ITERS_MAX_FRAC = 0.20         # maximum fraction of epochs with scaled lr

LINEAR_START_FACTOR_MIN = 0.05         # minimum starting lr factor
LINEAR_START_FACTOR_MAX = 0.9          # maximum starting lr factor
LINEAR_ITERS_MIN_FRAC = 0.01           # minimum fraction of epochs used for linear lr change
LINEAR_ITERS_MAX_FRAC = 0.30           # maximum fraction of epochs used for linear lr change

EXP_FINAL_FACTOR_MIN = 1e-3            # minimum final lr relative to initial lr
EXP_FINAL_FACTOR_MAX = 0.8             # maximum final lr relative to initial lr

COSINE_TMAX_MIN_FRAC = 0.50            # minimum fraction of epochs for one cosine decay cycle
COSINE_TMAX_MAX_FRAC = 1.00            # maximum fraction of epochs for one cosine decay cycle
COSINE_ETA_MIN_FACTOR_MIN = 1e-4       # minimum final lr relative to initial lr
COSINE_ETA_MIN_FACTOR_MAX = 1e-1       # maximum final lr relative to initial lr

PLATEAU_FACTOR_MIN = 0.1               # strongest lr reduction when validation loss stops improving
PLATEAU_FACTOR_MAX = 0.8               # weakest lr reduction when validation loss stops improving
PLATEAU_PATIENCE_MIN_FRAC = 0.02       # minimum fraction of epochs to wait before reducing lr
PLATEAU_PATIENCE_MAX_FRAC = 0.15       # maximum fraction of epochs to wait before reducing lr
PLATEAU_THRESHOLD_MIN = 1e-5           # minimum improvement considered significant
PLATEAU_THRESHOLD_MAX = 1e-3           # maximum improvement considered significant
PLATEAU_COOLDOWN_MAX_FRAC = 0.05       # maximum fraction of epochs to wait after an lr reduction

WARMRESTART_T0_MIN_FRAC = 0.05         # minimum fraction of epochs before the first restart
WARMRESTART_T0_MAX_FRAC = 0.30         # maximum fraction of epochs before the first restart
WARMRESTART_T_MULT = [1, 2, 3]         # multiplier applied to restart period after each restart
WARMRESTART_ETA_MIN_FACTOR_MIN = 1e-4  # minimum lr reached relative to initial lr
WARMRESTART_ETA_MIN_FACTOR_MAX = 1e-1  # maximum minimum lr relative to initial lr

POLYNOMIAL_ITERS_MIN_FRAC = 0.50       # minimum fraction of epochs over which polynomial decay acts
POLYNOMIAL_ITERS_MAX_FRAC = 1.00       # maximum fraction of epochs over which polynomial decay acts
POLYNOMIAL_POWER_MIN = 0.5             # minimum curvature of polynomial decay
POLYNOMIAL_POWER_MAX = 3.0             # maximum curvature of polynomial decay