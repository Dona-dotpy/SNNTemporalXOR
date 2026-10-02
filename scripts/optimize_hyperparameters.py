"""
Hyperparameter optimization script for SNN models using Optuna and W&B.

Main features:
- Optuna manages trials, sampling, pruning and persistent study storage in SQLite.
- W&B logs training/validation metrics for each trial and allows remote monitoring.
- Before resuming a previous optimization, the relevant parameters currently defined
  in `config.py` are compared with the configuration saved in
  `optimization_results/optimization_config.json`.
  If any parameter has changed, the differences are shown before deciding whether
  to restart the optimization from scratch.
- The target number of trials is treated as a total target, so interrupted studies
  can resume only the missing trials.
- Each trial periodically saves a PyTorch checkpoint containing at least epoch,
  model state, optimizer state, current best validation loss and trial parameters.
- If execution stops during a trial, training can resume from the latest checkpoint.
- The best model reached during each trial is saved separately from the resume checkpoint.
- Completed studies can optionally be extended with additional trials.
- The final script is intended to support conditional optimization of learning rate,
  optimizer, scheduler, loss function, initialization and SNN/model parameters.

Persistent data are separated into:
- Optuna database: study/trial state and objective values.
- W&B: monitoring and metric history.
- PyTorch checkpoints: actual training state required for resume.
"""

## --- LIBRARIES --- 

import optuna
import wandb
import os
import json
import sys
import shutil
from datetime import datetime
import time

import torch
import classes, functions, config

## --- FUNCTIONS ---

# store / load functions

def save_checkpoint(trial, epoch, model, optimizer, scheduler, lowest_loss):
    """ 
    Function useful to store a checkpoint, which means that we save all the relevant informations 
    that allow us to stop the execution of the hyperparameter optimization and start it again in
    a second moment

    Args:
        trial (optuna.trial.Trial):                                 current optuna trial 
        epoch (int):                                                last successfully completed training epoch
        model (torch.nn.Module):                                    model whose parameters must be saved
        optimizer (torch.optim.Optimizer):                          Optimizer associated with the current model
        scheduler (torch.optim.lr_scheduler.LRScheduler | None):    Scheduler associated with the optimizer
        lowest_loss (float):                                        Best validation loss reached so far during optimization

    Returns:
        None
    """

    # choose where to save the checkpoint
    checkpoint_dir = f"optimization_results/checkpoints/trial_{trial.number}"               # check for last trial performed
    os.makedirs(checkpoint_dir, exist_ok=True)                                              # create directory if not present
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")                         # join names to create full path

    # dictionary that contains all the relevant informations to start the optimization in a second moment
    checkpoint = {
        "trial_number": trial.number,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "lowest_loss": lowest_loss,
        "params": trial.params,
        "torch_rng_state": torch.get_rng_state(),           # state of the random number generator (to preserve deterministic behavior)
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(trial, model, optimizer, scheduler):
    """
    Load a previously stored checkpoint

    Args:
        trial (optuna.trial.Trial):                                Current Optuna trial
        model (torch.nn.Module):                                   Model where the saved parameters will be restored
        optimizer (torch.optim.Optimizer):                         Optimizer where the saved internal state will be restored
        cheduler (torch.optim.lr_scheduler.LRScheduler | None):    Scheduler associated with the optimizer

    Returns:
        tuple: 
            start_epoch (int):      first epoch that still needs to be executed
            lowest_loss (float):    Best validation loss reached before the interruption
            checkpoint (dict):      Complete checkpoint dictionary
    """

    # search and load checkpoint from its path if present, otherwise return None
    checkpoint_path = f"optimization_results/checkpoints/trial_{trial.number}/checkpoint.pt"  
    try:    
        checkpoint = torch.load(checkpoint_path, map_location = config.DEVICE)
    except FileNotFoundError:
        return None

    # assignment of checkpoint values when checkpoint is present
    model.load_state_dict(checkpoint["model_state_dict"])                   # load model paraemters 
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])           # load optimizer parameters   
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])       # load scheduler parameters 
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())                # get state of the random number generator to preserve deterministic behavior (on cpu because it refers to seed of cpu, not cuda)
    start_epoch = checkpoint["epoch"]+1                                     # select successive epoch from where we want to start again the optimization
    lowest_loss = checkpoint["lowest_loss"]                                 # load best result we obtained so far

    return start_epoch, lowest_loss, checkpoint 

def save_optimization_config(optimisation_config):
    """
    Save the current optimization configuration.

    Args:
        optimization_config (dict): Current hyperparameter optimization configuration

    Returns:
        None
    """
    os.makedirs("optimization_results", exist_ok = True)        # create the directory if not present

    # create/open the file .json to store the configuration and store the dictionary
    with open("optimization_results/optimization_config.json", "w") as file:    # "w" creates file if not present and, if present, overwrites the informations
        json.dump(optimisation_config, file, indent=4)                          # save the dictionary in the opened file with 4 indent spaces for readability

def get_optimization_config():
    """
    Looks into config.py in order to find the settings of the
    hyperparameter search. Useful to detect changes in the search space
    before resuming a previous optimization.

    Args:
        None

    Returns:
        dict: Hyperparameter search configuration currently defined in config.py
    """

    return {
        # General search space
        "lr_min": config.LR_MIN,
        "lr_max": config.LR_MAX,

        "hidden_layer_min": config.HIDDEN_LAYER_MIN,
        "hidden_layer_max": config.HIDDEN_LAYER_MAX,

        # Optimizer search space
        "optimizers": config.OPTIMIZERS,

        "weight_decay_min": config.WEIGHT_DECAY_MIN,
        "weight_decay_max": config.WEIGHT_DECAY_MAX,

        "beta1_min": config.BETA1_MIN,
        "beta1_max": config.BETA1_MAX,

        "beta2_min": config.BETA2_MIN,
        "beta2_max": config.BETA2_MAX,

        "momentum_min": config.MOMENTUM_MIN,
        "momentum_max": config.MOMENTUM_MAX,

        # Scheduler search space
        "schedulers": config.SCHEDULERS,

        "step_size_min_frac": config.STEP_SIZE_MIN_FRAC,
        "step_size_max_frac": config.STEP_SIZE_MAX_FRAC,
        "step_gamma_min": config.STEP_GAMMA_MIN,
        "step_gamma_max": config.STEP_GAMMA_MAX,

        "multistep_patterns": config.MULTISTEP_PATTERNS,
        "multistep_gamma_min": config.MULTISTEP_GAMMA_MIN,
        "multistep_gamma_max": config.MULTISTEP_GAMMA_MAX,

        "constant_factor_min": config.CONSTANT_FACTOR_MIN,
        "constant_factor_max": config.CONSTANT_FACTOR_MAX,
        "constant_iters_min_frac": config.CONSTANT_ITERS_MIN_FRAC,
        "constant_iters_max_frac": config.CONSTANT_ITERS_MAX_FRAC,

        "linear_start_factor_min": config.LINEAR_START_FACTOR_MIN,
        "linear_start_factor_max": config.LINEAR_START_FACTOR_MAX,
        "linear_iters_min_frac": config.LINEAR_ITERS_MIN_FRAC,
        "linear_iters_max_frac": config.LINEAR_ITERS_MAX_FRAC,

        "exp_final_factor_min": config.EXP_FINAL_FACTOR_MIN,
        "exp_final_factor_max": config.EXP_FINAL_FACTOR_MAX,

        "cosine_tmax_min_frac": config.COSINE_TMAX_MIN_FRAC,
        "cosine_tmax_max_frac": config.COSINE_TMAX_MAX_FRAC,
        "cosine_eta_min_factor_min": config.COSINE_ETA_MIN_FACTOR_MIN,
        "cosine_eta_min_factor_max": config.COSINE_ETA_MIN_FACTOR_MAX,

        "plateau_factor_min": config.PLATEAU_FACTOR_MIN,
        "plateau_factor_max": config.PLATEAU_FACTOR_MAX,
        "plateau_patience_min_frac": config.PLATEAU_PATIENCE_MIN_FRAC,
        "plateau_patience_max_frac": config.PLATEAU_PATIENCE_MAX_FRAC,
        "plateau_threshold_min": config.PLATEAU_THRESHOLD_MIN,
        "plateau_threshold_max": config.PLATEAU_THRESHOLD_MAX,
        "plateau_cooldown_max_frac": config.PLATEAU_COOLDOWN_MAX_FRAC,

        "warmrestart_t0_min_frac": config.WARMRESTART_T0_MIN_FRAC,
        "warmrestart_t0_max_frac": config.WARMRESTART_T0_MAX_FRAC,
        "warmrestart_t_mult": config.WARMRESTART_T_MULT,
        "warmrestart_eta_min_factor_min": config.WARMRESTART_ETA_MIN_FACTOR_MIN,
        "warmrestart_eta_min_factor_max": config.WARMRESTART_ETA_MIN_FACTOR_MAX,

        "polynomial_iters_min_frac": config.POLYNOMIAL_ITERS_MIN_FRAC,
        "polynomial_iters_max_frac": config.POLYNOMIAL_ITERS_MAX_FRAC,
        "polynomial_power_min": config.POLYNOMIAL_POWER_MIN,
        "polynomial_power_max": config.POLYNOMIAL_POWER_MAX,

        # Other hyperparameters
        "losses": config.LOSSES,
        "initializations": config.INITIALIZATIONS,

        # Training configuration that affects trial comparability
        "epochs_optimisation": config.EPOCHS_OPTIMISATION,
    }

def archive():
    """
    Archive the current Optuna database, optimization configuration and
    the whole checkpoints directory before starting a new optimization

    Args:
        None

    Returns:
        None
    """

    # create path where we want to archive everything and create directory 
    optimization_dir = "optimization_results"                           # directory where we save optimisations ended
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")            # get current data to creare directory result
    archive_dir = os.path.join(optimization_dir, "archive", timestamp)  # complete name
    os.makedirs(archive_dir, exist_ok = True)

    # check if the file path of the database exists and moves the database
    if os.path.exists(config.OPTUNA_DB_PATH):
        shutil.move(config.OPTUNA_DB_PATH, archive_dir)                 # move database from config.OPTUNA_DB_PATH to archive_dir     

    # check if the file path of the configuration file exists and moves it 
    config_path= os.path.join(optimization_dir, "optimization_config.json")
    if os.path.exists(config_path):
        shutil.move(config_path, archive_dir)                           # move configuration file from config_path to archive_dir               

    # create a path where we want to save old checkpoints and store them
    checkpoints_dir = os.path.join(optimization_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        shutil.move(checkpoints_dir, archive_dir)                       # move checkpoint file from checkpoint_dir to archive_dir

# optimizer functions

def define_optimizer(trial, model, lr):
    """
    Defines and generates the optimizer for the current Optuna trial.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial
        model (torch.nn.Module):    Model to which we want to apply the optimizer
        lr (float):                 Learning rate used by the optimizer

    Returns:
        torch.optim.Optimizer:      Optimizer generated with the sampled parameters
    """

    optimizer_name = trial.suggest_categorical("optimizer", config.OPTIMIZERS)
    use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
    if use_weight_decay:
        weight_decay = trial.suggest_float("weight_decay", config.WEIGHT_DECAY_MIN, config.WEIGHT_DECAY_MAX, log = True)
    else:
        weight_decay = 0.0
    if optimizer_name in ["Adam", "AdamW"]:
        beta1 = trial.suggest_float("beta1", config.BETA1_MIN, config.BETA1_MAX)
        beta2 = trial.suggest_float("beta2", config.BETA2_MIN, config.BETA2_MAX)
        ams_grad = trial.suggest_categorical("amsgrad", [True, False])
        momentum, nesterov = None, None
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", config.MOMENTUM_MIN, config.MOMENTUM_MAX)
        if momentum > 0:
            nesterov = trial.suggest_categorical("nesterov", [True, False])
        else:
            nesterov = False
        beta1, beta2, ams_grad = None, None, None

    if optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, beta2),
            weight_decay=weight_decay, amsgrad=ams_grad
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(beta1, beta2),
            weight_decay=weight_decay, amsgrad=ams_grad
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=nesterov
        )

def define_scheduler(trial, optimizer, lr):
    """
    Defines and generates the scheduler for the current Optuna trial.

    Args:
        trial (optuna.trial.Trial):      Current Optuna trial
        optimizer (torch.optim.Optimizer): Optimizer controlled by the scheduler
        lr (float):                     Initial learning rate

    Returns:
        torch.optim.lr_scheduler.LRScheduler | None:
            Scheduler generated with the sampled parameters
    """

    scheduler_name = trial.suggest_categorical(
        "scheduler",
        config.SCHEDULERS
    )

    epochs = config.EPOCHS_OPTIMISATION

    if scheduler_name == "None":
        return None

    elif scheduler_name == "StepLR":
        step_size = trial.suggest_int( "step_step_size", 
            max(1, int(config.STEP_SIZE_MIN_FRAC * epochs)), max(1, int(config.STEP_SIZE_MAX_FRAC * epochs))
        )
        gamma = trial.suggest_float("step_gamma", config.STEP_GAMMA_MIN, config.STEP_GAMMA_MAX)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,gamma=gamma)

    elif scheduler_name == "MultiStepLR":
        pattern = trial.suggest_categorical("multistep_pattern",config.MULTISTEP_PATTERNS)
        patterns = {"30_60": [0.30, 0.60], "50_75": [0.50, 0.75],
            "25_50_75": [0.25, 0.50, 0.75], "40_70_90": [0.40, 0.70, 0.90]
        }
        milestones = [int(epochs * fraction) for fraction in patterns[pattern]]
        gamma = trial.suggest_float("multistep_gamma", config.MULTISTEP_GAMMA_MIN, config.MULTISTEP_GAMMA_MAX)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=gamma)

    elif scheduler_name == "ConstantLR":

        factor = trial.suggest_float("constant_factor", config.CONSTANT_FACTOR_MIN, config.CONSTANT_FACTOR_MAX)
        total_iters = trial.suggest_int( "constant_total_iters", 
                max(1, int(config.CONSTANT_ITERS_MIN_FRAC * epochs)),max(1, int(config.CONSTANT_ITERS_MAX_FRAC * epochs))
        )
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=factor,total_iters=total_iters)

    elif scheduler_name == "LinearLR":

        start_factor = trial.suggest_float("linear_start_factor",
            config.LINEAR_START_FACTOR_MIN, config.LINEAR_START_FACTOR_MAX
        )
        total_iters = trial.suggest_int("linear_total_iters",
            max(1, int(config.LINEAR_ITERS_MIN_FRAC * epochs)), max(1, int(config.LINEAR_ITERS_MAX_FRAC * epochs))
        )
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=total_iters)

    elif scheduler_name == "ExponentialLR":

        final_factor = trial.suggest_float("exponential_final_factor", config.EXP_FINAL_FACTOR_MIN,
            config.EXP_FINAL_FACTOR_MAX, log=True
        )
        gamma = final_factor ** (1 / epochs)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_name == "CosineAnnealingLR":

        t_max = trial.suggest_int("cosine_t_max",
            max(1, int(config.COSINE_TMAX_MIN_FRAC * epochs)), max(1, int(config.COSINE_TMAX_MAX_FRAC * epochs))
        )
        eta_min_factor = trial.suggest_float( "cosine_eta_min_factor", config.COSINE_ETA_MIN_FACTOR_MIN, 
                            config.COSINE_ETA_MIN_FACTOR_MAX, log=True
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr * eta_min_factor)

    elif scheduler_name == "ReduceLROnPlateau":

        factor = trial.suggest_float("plateau_factor",
            config.PLATEAU_FACTOR_MIN, config.PLATEAU_FACTOR_MAX
        )
        patience = trial.suggest_int( "plateau_patience",
            max(1, int(config.PLATEAU_PATIENCE_MIN_FRAC * epochs)), max(1, int(config.PLATEAU_PATIENCE_MAX_FRAC * epochs))
        )
        threshold = trial.suggest_float("plateau_threshold", config.PLATEAU_THRESHOLD_MIN,
            config.PLATEAU_THRESHOLD_MAX, log=True
        )
        cooldown = trial.suggest_int("plateau_cooldown",
            0, max(1, int(config.PLATEAU_COOLDOWN_MAX_FRAC * epochs))
        )

        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, threshold=threshold, cooldown=cooldown)

    elif scheduler_name == "CosineAnnealingWarmRestarts":
        t_0 = trial.suggest_int("warmrestart_t0", max(1, int(config.WARMRESTART_T0_MIN_FRAC * epochs)),
            max(1, int(config.WARMRESTART_T0_MAX_FRAC * epochs))
        )
        t_mult = trial.suggest_categorical("warmrestart_t_mult", config.WARMRESTART_T_MULT)
        eta_min_factor = trial.suggest_float("warmrestart_eta_min_factor", config.WARMRESTART_ETA_MIN_FACTOR_MIN,
            config.WARMRESTART_ETA_MIN_FACTOR_MAX, log=True
        )
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=lr * eta_min_factor)

    elif scheduler_name == "PolynomialLR":

        total_iters = trial.suggest_int("polynomial_total_iters", max(1, int(config.POLYNOMIAL_ITERS_MIN_FRAC * epochs)),
            max(1, int(config.POLYNOMIAL_ITERS_MAX_FRAC * epochs))
        )
        power = trial.suggest_float("polynomial_power",
            config.POLYNOMIAL_POWER_MIN, config.POLYNOMIAL_POWER_MAX
        )
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=power)

# Objective functions

def objective(trial, train_loader, val_loader, start_time, previous_optimization_time):
    """
    Objective function used by Optuna.

    This function is executed once for each trial of the optimization study.
    The `trial` object is created and passed automatically by Optuna.

    Inside this function, hyperparameters are defined through `trial.suggest_*`
    methods. Each call defines one dimension of the search space.

    Main methods:

    - trial.suggest_float(name, low, high)
        Suggests a floating-point value in the interval [low, high].

        Example:
            lr = trial.suggest_float("lr", 1e-5, 1e-2)

    - trial.suggest_float(name, low, high, log=True)
        Suggests a floating-point value on a logarithmic scale.
        Useful when the parameter spans multiple orders of magnitude.

        Typical use cases:
            learning rate
            weight decay
            regularization coefficients

        Example:
            lr = trial.suggest_float(
                "lr",
                1e-5,
                1e-2,
                log=True
            )

    - trial.suggest_int(name, low, high)
        Suggests an integer value in the interval [low, high].

        Typical use cases:
            step_size
            T_max
            number of neurons
            number of layers

        Example:
            step_size = trial.suggest_int(
                "step_size",
                10,
                300
            )

    - trial.suggest_categorical(name, choices)
        Selects one value from a predefined set of alternatives.

        Typical use cases:
            optimizer
            scheduler
            loss function
            weight initialization method

        Example:
            optimizer_name = trial.suggest_categorical(
                "optimizer",
                ["Adam", "AdamW", "SGD"]
            )

    Important concepts:

    - The first argument of every `suggest_*` method is the name used by Optuna
      to identify and store that hyperparameter.

    - Each trial receives one specific combination of suggested values.

    - All `suggest_*` calls together define the hyperparameter search space.

    - The function must return the objective value that Optuna has to optimize.

        Example:
            return validation_loss

    - If the Study was created with:
            direction="minimize"
      Optuna searches for the hyperparameters that produce the lowest returned
      objective value.

    - If the Study was created with:
            direction="maximize"
      Optuna searches for the hyperparameters that produce the highest returned
      objective value.

    Example search space:

        lr = trial.suggest_float(
            "lr",
            1e-5,
            1e-2,
            log=True
        )

        optimizer = trial.suggest_categorical(
            "optimizer",
            ["Adam", "AdamW"]
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-8,
            1e-3,
            log=True
        )

    A possible trial could therefore be:

        lr = 0.00073
        optimizer = "AdamW"
        weight_decay = 2.1e-5

    The final returned value is the score associated with that specific
    hyperparameter configuration.
    """

    torch.manual_seed(config.SEED)      # seed setup for reproducibility 

    # hyperparameter sampling and
    # optimizer parameters search -> we don't look only to "which optimizer" but to "which optimizer and with which parameters"!

    hidden_layer = trial.suggest_int("hidden_layer", config.HIDDEN_LAYER_MIN, config.HIDDEN_LAYER_MAX)
    model = classes.XORSNN(hidden_layer=hidden_layer).to(config.DEVICE)
    lr = trial.suggest_float("lr", config.LR_MIN, config.LR_MAX, log=True)                          # we look for a float search in logaritmic scale (big interval)
    optimizer = define_optimizer(trial, model, lr)
    scheduler = define_scheduler(trial, optimizer, lr)
    loss_fn = torch.nn.CrossEntropyLoss()
         
    run = None                          # to avoid any problem with "finally"          

    # use try/finally structure to ensure closure of the wandb communication in case of any error
    try: 

        # setup a communication with W&B (previous login from terminal is required)
        run = wandb.init(                                                                               
                project="SNNTemporalXOR",                                                               # name of the project we will see on W&B
                group="hyperparameter_optimization",                                                     # every run will be under the same group to record data
                name=f"trial-{trial.number}",                                                           
                config={                                                                                # fixed parameters of the run
                    "trial": trial.number,
                    "lr": lr,
                    "epochs": config.EPOCHS_OPTIMISATION
                },
                reinit="finish_previous"                                                                # magaes the situation in which there is another run active (waits the previuos one to finish)
            )
        

        # checkpoint load
        checkpoint = load_checkpoint(trial, model, optimizer, scheduler)
        if checkpoint is None:  
            start_epoch = 0
            lowest_loss = float("inf")                                                                  # generate +inf since we are in front of a minimization problem
        else:
            start_epoch, lowest_loss, checkpoint = checkpoint

        # train/validate loop
        for epoch in range(start_epoch, config.EPOCHS_OPTIMISATION):                                   
            avg_loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)        # train function
            avg_loss_val, acc_val = functions.evaluate(model, val_loader, loss_fn)                      # validation function

            # scheduler update, ReduceLROnPlateau requires a different step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss_val)
                else:
                    scheduler.step()

            # keep the lowest validation loss reached during training, not only the final one
            if avg_loss_val < lowest_loss:                                                              
                lowest_loss = avg_loss_val

            # decide which data we have to send to W&B (data necessary to build graphs and metrics inside the app)
            current_sim_time = previous_optimization_time + (time.perf_counter() - start_time)      # total simulation time up to now
            study.set_user_attr("optimization_time", current_sim_time)
            # convert simulation time into [hh:mm:ss]
            hours = int(current_sim_time// 3600)                                   # // on a float returns a float
            minutes = int((current_sim_time - hours*3600) // 60)
            seconds = int(current_sim_time % 60)
            sim_time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}" 
            run.log({                                                                                   
                "epoch": epoch,
                "train_loss": avg_loss_train,
                "val_loss": avg_loss_val,
                "train_accuracy": acc_train,
                "val_accuracy": acc_val,
                "lr": optimizer.param_groups[0]["lr"],
                "sim_time": current_sim_time
            })
            # decide which data the user sees in the inital summary
            run.summary["best_val_loss"] = lowest_loss
            run.summary["final_val_loss"] = avg_loss_val
            run.summary["final_val_accuracy"] = acc_val
            run.summary["completed_epochs"] = epoch + 1
            run.summary["final_lr"] = optimizer.param_groups[0]["lr"]
            run.summary["status"] = "in progress" 
            run.summary["simulation_time"] = sim_time_string                # refresh summary table on W&B

            # save the checkpoint every config.CHECKPOINT_INTERVAL epochs
            if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(trial, epoch, model, optimizer, scheduler, lowest_loss)

            trial.report(avg_loss_val, step=epoch)                          # reports the current validation loss to Optuna study for this epoch  

            # check if it's convenient an early stop
            if trial.should_prune():
                run.summary["status"] = "pruned"                                                                    
                raise optuna.TrialPruned()
            
        run.summary["status"] = "completed"                             # if the for cicle ended correctly, the trial is completed 
    
    finally:
        if run is not None:
            run.finish()            # interrupt W&B transmission
    
    return lowest_loss

## --- MAIN CODE ---

# get optimization configurations: current and saved ones
current_config = get_optimization_config()
config_path = "optimization_results/optimization_config.json"
if os.path.exists(config_path):
    with open("optimization_results/optimization_config.json", "r") as file:
        saved_config = json.load(file)
else:
    saved_config = None

# generate train and validation dataset deterministaticalli
train_dataset, val_dataset = functions.generate_train_validation_dataset()
train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)

# check if a study is already present
study_exists = os.path.exists(config.OPTUNA_DB_PATH)

if not study_exists:
    print("\nNo optimization present, let's start a new one from scratch...\n")
    save_optimization_config(current_config)                                        # config.py script could have been changed

elif current_config == saved_config:                                                # No change in the configuration file
    while True:
        selection = input("\n[y/n] Previous optimization found, do you want to continue?\n").strip().lower()
        if selection == "y":
            break
        elif selection == "n":
            sys.exit()                                                              # end the the execution of the script
        print("Input not recognized. Please insert 'y' for yes and 'n' for no")

else:
    while True:
        selection = input("\n[y/n] Configurations changed, do you want to start a new optimization?\n").strip().lower()
        if selection == "y":
            archive()
            save_optimization_config(current_config)
            break
        elif selection == "n":
            sys.exit()
        print("Input not recognized. Please insert 'y' for yes and 'n' for no")

# configuration of the sampler and of the pruner
sampler = optuna.samplers.TPESampler(seed = config.SEED)                        # decides which values oh hyperparameters we have to try in the next trial, in this case it's based on best results of previus trials
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)    # for a minimization study, prunes trials whose intermediate loss is worse than the median of previous completed trials (1st 5 trials completed interely, never stop before epoch 100)

# load/create the study to run the optimization alogithm
study = optuna.create_study(                                       
    study_name = config.STUDY_NAME,
    direction="minimize",                               # task required
    sampler = sampler,
    pruner = pruner,
    storage=f"sqlite:///{config.OPTUNA_DB_PATH}",       # storage location
    load_if_exists=True                                 # if a study is already present, do now create a new one but load the existing one
    )

# initialize time to the initial value
previous_optimization_time = study.user_attrs.get("optimization_time", 0.0)     # obtain optimization time from previuos optimization
optimization_start_time = time.perf_counter()                                   # reference to time from time library

# Compute the trial reamining before the end of the study (useful when we load a checkpoint)
remaining_trials = config.TARGET_TRIALS - len(study.trials)
if remaining_trials > 0:
    study.optimize(                                                     # we pass to the study object what we must minimize and in how many trials
        lambda trial: objective(trial, train_loader, val_loader, optimization_start_time, previous_optimization_time),       # Optuna expects a function callable as f(trial); we map objective(trial, train_ld, test_ld) -> f(trial)
        n_trials=remaining_trials
    )                               

# Values we want to print in the terminal
print("Best value:", study.best_value)
print("Best parameters:", study.best_params)
