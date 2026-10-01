"""
Hyperparameter optimization script for SNN models using Optuna and W&B.

Main features:
- Optuna manages trials, sampling, pruning and persistent study storage in SQLite.
- W&B logs training/validation metrics for each trial and allows remote monitoring.
- Before resuming a previous optimization, the relevant parameters currently defined
  in `config.py` are compared with the configuration saved in
  `optimization_results/hyperparameter_search/optimization_config.json`.
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

import torch
import classes, functions, config

## --- FUNCTIONS ---

def save_checkpoint(trial, epoch, model, optimizer, lowest_loss):
    """ 
    Function useful to store a checkpoint, which means that we save all the relevant informations 
    that allow us to stop the execution of the hyperparameter optimization and start it again in
    a second moment

    Args:
        trial (optuna.trial.Trial):         current optuna trial 
        epoch (int):                        last successfully completed training epoch
        model (torch.nn.Module):            model whose parameters must be saved
        optimizer (torch.optim.Optimizer):  Optimizer associated with the current model
        lowest_loss (float):                Best validation loss reached so far during optimization

    Returns:
        None
    """

    # choose where to save the checkpoint
    checkpoint_dir = f"optimization_results/checkpoints/trial_{trial.number}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

    # dictionary that contains all the relevant informations to start the optimization in a second moment
    checkpoint = {
        "trial_number": trial.number,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lowest_loss": lowest_loss,
        "params": trial.params,
        "torch_rng_state": torch.get_rng_state(),           # state of the random number generator (to preserve deterministic behavior)
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(trial, model, optimizer):
    """
    Load a previously stored checkpoint

    Args:
        trial (optuna.trial.Trial):          Current Optuna trial
        model (torch.nn.Module):             Model where the saved parameters will be restored
        optimizer (torch.optim.Optimizer):   Optimizer where the saved internal state will be restored

    Returns:
        tuple: 
            start_epoch (int):      first epoch that still needs to be executed
            lowest_loss (float):    Best validation loss reached before the interruption
            checkpoint (dict):      Complete checkpoint dictionary
    """

    # search and load checkpoint, if present, otherwise return None
    checkpoint_path = f"optimization_results/checkpoints/trial_{trial.number}/checkpoint.pt"  
    try:    
        checkpoint = torch.load(checkpoint_path, map_location = config.DEVICE)
    except FileNotFoundError:
        return None

    # assignment of checkpoint values
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])       
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())            # get state of the random number generator to preserve deterministic behavior (on cpu because it refers to seed of cpu, not cuda)
    start_epoch = checkpoint["epoch"]+1
    lowest_loss = checkpoint["lowest_loss"]

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
    functions that looks into config.py script in order to find the settings
    of the hyperparameter search, useful to compare any changes with the
    checkpoint

    Args:
        None

    Returns:
        dict: Hyperparameter search configuration currently defined in config.py
    """

    return {
        "lr_min": config.LR_MIN,
        "lr_max": config.LR_MAX,

        "beta_min": config.BETA_MIN,
        "beta_max": config.BETA_MAX,

        "hidden_layer_min": config.HIDDEN_LAYER_MIN,
        "hidden_layer_max": config.HIDDEN_LAYER_MAX,

        "optimizers": config.OPTIMIZERS,
        "schedulers": config.SCHEDULERS,
        "losses": config.LOSSES,
        "initializations": config.INITIALIZATIONS,
    }

def archive_optimization():
    """
    Archive the current optimization before starting a new one

    Args:
        None

    Returns:
        None
    """

    # create path where we want to save old optimizations and create directory 
    optimization_dir = "optimization_results"                           # directory where we save optimisations ended
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")            # get current data to creare directory result
    archive_dir = os.path.join(optimization_dir, "archive", timestamp)  # complete name
    os.makedirs(archive_dir, exist_ok = True)

    # check if the file path exists before moving it
    if os.path.exists(config.OPTUNA_DB_PATH):
        shutil.move(config.OPTUNA_DB_PATH, archive_dir)

    # check if the file path exists before moving it 
    config_path = os.path.join(optimization_dir, "optimization_config.json")
    if os.path.exists(config_path):
        shutil.move(config_path, archive_dir)

    # create a path where we want to save old checkpoints and create directory
    checkpoints_dir = os.path.join(optimization_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        shutil.move(checkpoints_dir, archive_dir)

def objective(trial, train_loader, val_loader):
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
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)                                            # we look for a float searchd in logaritmic scale
    torch.manual_seed(config.SEED)                                                                  # set seed for reproducibility

    run = wandb.init(                                                                               # initialize weights and biases to see training steps on W&B
            project="SNNTemporalXOR",
            group="learning_rate_optimization",
            name=f"trial-{trial.number}",
            config={
                "trial": trial.number,
                "lr": lr,
                "epochs": config.EPOCHS_OPTIMISATION
            },
            reinit="finish_previous"
        )

    # use try/finally structure to ensure closure of the wandb communication in case of any error
    try:        
        # parameters of the net and optimizators
        model = classes.XORSNN().to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        checkpoint = load_checkpoint(trial, model, optimizer)
        if checkpoint is None:
            start_epoch = 0
            lowest_loss = float("inf")                                                                  # generate +inf since we are in front of a minimization problem
        else:
            start_epoch, lowest_loss, checkpoint = checkpoint

        for epoch in range(start_epoch, config.EPOCHS_OPTIMISATION):                                                              # train loop
            avg_loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)
            avg_loss_val, acc_val = functions.evaluate(model, val_loader, loss_fn)

            run.log({                                                                                   # send data to iPhone
                "epoch": epoch,
                "train_loss": avg_loss_train,
                "val_loss": avg_loss_val,
                "train_accuracy": acc_train,
                "val_accuracy": acc_val,
                "lr": optimizer.param_groups[0]["lr"]
            })          

            # keep the lowest validation loss reached during training, not only the final one
            if avg_loss_val < lowest_loss:                                                              
                lowest_loss = avg_loss_val

            # save the checkpoint every 10 epochs
            if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(trial, epoch, model, optimizer, lowest_loss)

            trial.report(avg_loss_val, step=epoch)                                                      # reports the current validation loss to Optuna for this epoch  

            # check if it's convenient an early stop
            if trial.should_prune():                                                                    
                raise optuna.TrialPruned()

    finally:
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

study_exists = os.path.exists(config.OPTUNA_DB_PATH)

if not study_exists:
    print("\nNo optimization present, let's start a new one from scratch...\n")
    save_optimization_config(current_config)

elif current_config == saved_config:
    while True:
        selection = input("\n[y/n] Previous optimization found, do you want to continue?\n").strip().lower()
        if selection == "y":
            break
        elif selection == "n":
            sys.exit()
        print("Input not recognized. Please insert 'y' for yes and 'n' for no")

else:
    while True:
        selection = input("\n[y/n] Configurations changed, do you want to start a new optimization?\n").strip().lower()
        if selection == "y":
            archive_optimization()
            save_optimization_config(current_config)
            break
        elif selection == "n":
            sys.exit()
        print("Input not recognized. Please insert 'y' for yes and 'n' for no")

sampler = optuna.samplers.TPESampler(seed = config.SEED)                        # decides which values oh hyperparameters we have to try in the next trial, in this case it's based on best results of previus trials
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)    # for a minimization study, prunes trials whose intermediate loss is worse than the median of previous completed trials (1st 5 trials completed interely, never stop before epoch 100)

study = optuna.create_study(                                        # creates a new optimization study aimed at minimizing the objective value and stores it in a Study object that manages the trials
    study_name = config.STUDY_NAME,
    direction="minimize",
    sampler = sampler,
    pruner = pruner,
    storage=f"sqlite:///{config.OPTUNA_DB_PATH}",
    load_if_exists=True
    )

remaining_trials = config.TARGET_TRIALS - len(study.trials)
if remaining_trials > 0:
    study.optimize(                                                     # we pass to the study object what we must minimize and in how many trials
        lambda trial: objective(trial, train_loader, val_loader),       # Optuna expects a function callable as f(trial); we map objective(trial, train_ld, test_ld) -> f(trial)
        n_trials=remaining_trials
    )                               

print("Best value:", study.best_value)
print("Best parameters:", study.best_params)


