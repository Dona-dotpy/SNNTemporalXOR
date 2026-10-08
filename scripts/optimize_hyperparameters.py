"""
Config-driven Hyperparameter Optimization engine for SNN models using Optuna and W&B.

Architecture & Pipeline:
- Driven entirely by JSON experiment files located in `optimization/inputs/<study_name>.json`.
- Automatically initializes and isolates study environments under `optimization/studies/<study_name>/`:
    * Persistent SQLite database: `<study_name>.db`
    * Configuration snapshot: `config.json`
    * Trial checkpoints: `checkpoints/trial_<number>/checkpoint.pt`
    * Best checkpointed models: `best_models/trial_<number>/best_model.pt`
- Prioritizes `fixed_params` over `search_space`, bypassing Optuna sampling for pinned values
  to avoid wasting trials on known hyperparameters.
- Saves PyTorch training checkpoints (model, optimizer, scheduler, RNG state) for seamless resume.
- Synchronizes trial metrics, execution time, and complete parameter configurations with W&B.
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
import argparse                                                                                 # CLI argument parser

import torch
import classes, functions, config

## --- FUNCTIONS ---

# store / load functions

def save_checkpoint(trial, epoch, model, optimizer, scheduler, lowest_loss, checkpoints_dir):
    """ 
    Saves a training checkpoint into the study-specific directory to resume later.

    Args:
        trial (optuna.trial.Trial):                                 Current optuna trial 
        epoch (int):                                                Last completed training epoch
        model (torch.nn.Module):                                    Model whose parameters must be saved
        optimizer (torch.optim.Optimizer):                          Optimizer associated with the model
        scheduler (torch.optim.lr_scheduler.LRScheduler | None):    Scheduler associated with the optimizer
        lowest_loss (float):                                        Best validation loss reached so far
        checkpoints_dir (str):                                      Study-specific checkpoints root directory

    Returns:
        None
    """
    checkpoint_dir = os.path.join(checkpoints_dir, f"trial_{trial.number}")                     # trial-specific checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)                                                  # ensure directory exists
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")                             # full checkpoint file path

    checkpoint = {
        "trial_number": trial.number,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "lowest_loss": lowest_loss,
        "params": trial.params,
        "torch_rng_state": torch.get_rng_state(),                                               # preserve deterministic RNG
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(trial, model, optimizer, scheduler, checkpoints_dir):
    """
    Loads a previously stored checkpoint from the study-specific directory.

    Args:
        trial (optuna.trial.Trial):                                 Current Optuna trial
        model (torch.nn.Module):                                    Model where parameters will be restored
        optimizer (torch.optim.Optimizer):                          Optimizer where state will be restored
        scheduler (torch.optim.lr_scheduler.LRScheduler | None):    Scheduler associated with the optimizer
        checkpoints_dir (str):                                      Study-specific checkpoints root directory

    Returns:
        tuple | None: (start_epoch, lowest_loss, checkpoint) if found, else None
    """
    checkpoint_path = os.path.join(checkpoints_dir, f"trial_{trial.number}", "checkpoint.pt") # locate checkpoint file
    try:    
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    except FileNotFoundError:
        return None                                                                             # no previous checkpoint found

    model.load_state_dict(checkpoint["model_state_dict"])                                       # restore model weights
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])                               # restore optimizer state
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])                           # restore scheduler state
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())                                    # restore CPU RNG state
    start_epoch = checkpoint["epoch"] + 1                                                       # resume from next epoch
    lowest_loss = checkpoint["lowest_loss"]                                                     # best loss so far

    return start_epoch, lowest_loss, checkpoint

# study functions 

## --- CONFIGURATION HELPERS ---

def load_study_config(config_name_or_path):
    """Loads study configuration JSON from optimization/inputs/ or custom path."""
    if not os.path.exists(config_name_or_path):
        candidate = os.path.join("optimization", "inputs", config_name_or_path)                 # check inputs directory
        if os.path.exists(candidate):
            config_name_or_path = candidate
        elif not config_name_or_path.endswith(".json"):
            candidate = os.path.join("optimization", "inputs", f"{config_name_or_path}.json")  # append .json if omitted
            if os.path.exists(candidate):
                config_name_or_path = candidate
    with open(config_name_or_path, "r") as file:
        return json.load(file)                                                                  # load study config dictionary

def build_sampler(sampler_cfg):
    """Instantiates the Optuna sampler specified in the study configuration."""
    s_type = sampler_cfg.get("type", "TPESampler")
    seed = sampler_cfg.get("seed", config.SEED)
    if s_type == "TPESampler":
        return optuna.samplers.TPESampler(seed=seed)                                            # TPE Bayesian sampler
    elif s_type == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=seed)                                         # uniform random sampler
    raise ValueError(f"Unknown sampler type: {s_type}")

def build_pruner(pruner_cfg):
    """Instantiates the Optuna pruner specified in the study configuration."""
    p_type = pruner_cfg.get("type", "MedianPruner")
    if p_type == "MedianPruner":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 20),                            # trials before pruning starts
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 300)                                # min epochs before trial pruning
        )
    elif p_type == "NopPruner":
        return optuna.pruners.NopPruner()                                                       # no pruning
    raise ValueError(f"Unknown pruner type: {p_type}")

# optimizer functions

def define_optimizer(trial, model, lr, fixed_params, search_space):
    """Builds optimizer using fixed parameters when present or sampling from search space."""
    optimizer_name = fixed_params.get("optimizer") or trial.suggest_categorical("optimizer", search_space["optimizers"])

    if "weight_decay" in fixed_params:
        weight_decay = fixed_params["weight_decay"]
    else:
        use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
        weight_decay = trial.suggest_float("weight_decay", search_space["weight_decay_min"], search_space["weight_decay_max"], log=True) if use_weight_decay else 0.0

    if optimizer_name in ["Adam", "AdamW"]:
        beta1 = fixed_params.get("beta1") if "beta1" in fixed_params else trial.suggest_float("beta1", search_space["beta1_min"], search_space["beta1_max"])
        beta2 = fixed_params.get("beta2") if "beta2" in fixed_params else trial.suggest_float("beta2", search_space["beta2_min"], search_space["beta2_max"])
        ams_grad = fixed_params.get("amsgrad") if "amsgrad" in fixed_params else trial.suggest_categorical("amsgrad", [True, False])
        momentum, nesterov = None, None
    elif optimizer_name == "SGD":
        momentum = fixed_params.get("momentum") if "momentum" in fixed_params else trial.suggest_float("momentum", search_space["momentum_min"], search_space["momentum_max"])
        nesterov = (fixed_params.get("nesterov") if "nesterov" in fixed_params else trial.suggest_categorical("nesterov", [True, False])) if momentum > 0 else False
        beta1, beta2, ams_grad = None, None, None

    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, amsgrad=ams_grad)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, amsgrad=ams_grad)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

def define_scheduler(trial, optimizer, lr, epochs, fixed_params, search_space):
    """Builds LR scheduler using fixed parameters when present or sampling from search space."""
    schedulers_pool = search_space.get("schedulers", ["None"])
    scheduler_name = fixed_params.get("scheduler") or (trial.suggest_categorical("scheduler", schedulers_pool) if len(schedulers_pool) > 1 else schedulers_pool[0])

    if scheduler_name == "None":
        return None
    elif scheduler_name == "StepLR":
        step_size = fixed_params.get("step_step_size") if "step_step_size" in fixed_params else trial.suggest_int("step_step_size", max(1, int(search_space["step_size_min_frac"] * epochs)), max(1, int(search_space["step_size_max_frac"] * epochs)))
        gamma = fixed_params.get("step_gamma") if "step_gamma" in fixed_params else trial.suggest_float("step_gamma", search_space["step_gamma_min"], search_space["step_gamma_max"])
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "MultiStepLR":
        patterns = search_space.get("multistep_patterns", ["25_50_75"])
        pattern = fixed_params.get("multistep_pattern") or (trial.suggest_categorical("multistep_pattern", patterns) if len(patterns) > 1 else patterns[0])
        pattern_map = {"30_60": [0.30, 0.60], "50_75": [0.50, 0.75], "25_50_75": [0.25, 0.50, 0.75], "40_70_90": [0.40, 0.70, 0.90]}
        milestones = [int(epochs * f) for f in pattern_map[pattern]]
        gamma = fixed_params.get("multistep_gamma") if "multistep_gamma" in fixed_params else trial.suggest_float("multistep_gamma", search_space["multistep_gamma_min"], search_space["multistep_gamma_max"])
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == "LinearLR":
        start_factor = fixed_params.get("linear_start_factor") if "linear_start_factor" in fixed_params else trial.suggest_float("linear_start_factor", search_space["linear_start_factor_min"], search_space["linear_start_factor_max"])
        total_iters = fixed_params.get("linear_total_iters") if "linear_total_iters" in fixed_params else trial.suggest_int("linear_total_iters", max(1, int(search_space["linear_iters_min_frac"] * epochs)), max(1, int(search_space["linear_iters_max_frac"] * epochs)))
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=total_iters)
    elif scheduler_name == "ConstantLR":
        factor = fixed_params.get("constant_factor") if "constant_factor" in fixed_params else trial.suggest_float("constant_factor", search_space["constant_factor_min"], search_space["constant_factor_max"])
        total_iters = fixed_params.get("constant_total_iters") if "constant_total_iters" in fixed_params else trial.suggest_int("constant_total_iters", max(1, int(search_space["constant_iters_min_frac"] * epochs)), max(1, int(search_space["constant_iters_max_frac"] * epochs)))
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=factor, total_iters=total_iters)
    elif scheduler_name == "ExponentialLR":
        final_factor = fixed_params.get("exponential_final_factor") if "exponential_final_factor" in fixed_params else trial.suggest_float("exponential_final_factor", search_space["exp_final_factor_min"], search_space["exp_final_factor_max"], log=True)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=final_factor ** (1 / epochs))
    elif scheduler_name == "CosineAnnealingLR":
        eta_min_factor = fixed_params.get("cosine_eta_min_factor") if "cosine_eta_min_factor" in fixed_params else trial.suggest_float("cosine_eta_min_factor", search_space["cosine_eta_min_factor_min"], search_space["cosine_eta_min_factor_max"], log=True)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * eta_min_factor)
    elif scheduler_name == "ReduceLROnPlateau":
        factor = fixed_params.get("plateau_factor") if "plateau_factor" in fixed_params else trial.suggest_float("plateau_factor", search_space["plateau_factor_min"], search_space["plateau_factor_max"])
        patience = fixed_params.get("plateau_patience") if "plateau_patience" in fixed_params else trial.suggest_int("plateau_patience", max(1, int(search_space["plateau_patience_min_frac"] * epochs)), max(1, int(search_space["plateau_patience_max_frac"] * epochs)))
        threshold = fixed_params.get("plateau_threshold") if "plateau_threshold" in fixed_params else trial.suggest_float("plateau_threshold", search_space["plateau_threshold_min"], search_space["plateau_threshold_max"], log=True)
        cooldown = fixed_params.get("plateau_cooldown") if "plateau_cooldown" in fixed_params else trial.suggest_int("plateau_cooldown", 0, max(1, int(search_space["plateau_cooldown_max_frac"] * epochs)))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, threshold=threshold, cooldown=cooldown)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        t_0 = fixed_params.get("warmrestart_t0") if "warmrestart_t0" in fixed_params else trial.suggest_int("warmrestart_t0", max(1, int(search_space["warmrestart_t0_min_frac"] * epochs)), max(1, int(search_space["warmrestart_t0_max_frac"] * epochs)))
        t_mult = fixed_params.get("warmrestart_t_mult") if "warmrestart_t_mult" in fixed_params else trial.suggest_categorical("warmrestart_t_mult", search_space["warmrestart_t_mult"])
        eta_min_factor = fixed_params.get("warmrestart_eta_min_factor") if "warmrestart_eta_min_factor" in fixed_params else trial.suggest_float("warmrestart_eta_min_factor", search_space["warmrestart_eta_min_factor_min"], search_space["warmrestart_eta_min_factor_max"], log=True)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=lr * eta_min_factor)
    elif scheduler_name == "PolynomialLR":
        total_iters = fixed_params.get("polynomial_total_iters") if "polynomial_total_iters" in fixed_params else trial.suggest_int("polynomial_total_iters", max(1, int(search_space["polynomial_iters_min_frac"] * epochs)), max(1, int(search_space["polynomial_iters_max_frac"] * epochs)))
        power = fixed_params.get("polynomial_power") if "polynomial_power" in fixed_params else trial.suggest_float("polynomial_power", search_space["polynomial_power_min"], search_space["polynomial_power_max"])
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=power)

def define_initialization(trial, model, input_activity, fixed_params, search_space):
    """Initializes network weights with priority to fixed_params or sampling from search_space."""
    inits = search_space.get("initializations", ["default"])
    init_name = fixed_params.get("initialization") or (trial.suggest_categorical("initialization", inits) if len(inits) > 1 else inits[0])
    if init_name == "default":
        return

    if init_name in ["xavier_uniform", "xavier_normal"]:
        gain = fixed_params.get("xavier_gain") if "xavier_gain" in fixed_params else trial.suggest_float("xavier_gain", search_space["xavier_gain_min"], search_space["xavier_gain_max"])
    elif init_name == "orthogonal":
        gain = fixed_params.get("orthogonal_gain") if "orthogonal_gain" in fixed_params else trial.suggest_float("orthogonal_gain", search_space["orthogonal_gain_min"], search_space["orthogonal_gain_max"])
    elif init_name == "fluctuation_driven":
        xi = fixed_params.get("fluct_xi") if "fluct_xi" in fixed_params else trial.suggest_float("fluct_xi", search_space["fluct_xi_min"], search_space["fluct_xi_max"])

    for i, module in enumerate(model.net):
        if isinstance(module, torch.nn.Linear):
            if init_name == "xavier_uniform":
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_name == "xavier_normal":
                torch.nn.init.xavier_normal_(module.weight, gain=gain)
            elif init_name == "orthogonal":
                torch.nn.init.orthogonal_(module.weight, gain=gain)
            elif init_name == "fluctuation_driven":
                lif_layer = model.net[i + 1]
                threshold, beta = lif_layer.threshold.item(), config.BETA
                sigma_u = threshold / xi                                                                # membrane fluctuation target std
                epsilon_hat = 1.0 / (1.0 - beta ** 2)                                                   # discrete LIF decay kernel sum
                weight_std = sigma_u / (module.in_features * input_activity * epsilon_hat) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=weight_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

def compute_input_activity(train_loader):
    """
    Computes the average input activity over the training dataset.

    Args:
        train_loader (torch.utils.data.DataLoader): Training dataloader

    Returns:
        float: Average probability of an input being active at one time step
    """

    total_activity = 0.0
    total_elements = 0

    for data, _ in train_loader:
        total_activity += data.sum().item()
        total_elements += data.numel()

    return total_activity / total_elements

# Objective functions

def objective(trial, study_config, study, train_loader, val_loader, input_activity, checkpoints_dir, best_models_dir, start_time, previous_optimization_time):
    """
    Objective function executed for each Optuna trial.
    Reads experiment settings from study_config, builds model, trains and evaluates.
    """
    torch.manual_seed(study_config.get("seed", config.SEED))                                    # trial seed setup

    fixed_params = study_config.get("fixed_params", {})                                        # constant experiment parameters
    search_space = study_config.get("search_space", {})                                        # search intervals and choices
    epochs = study_config.get("epochs", 1000)                                                  # total training epochs

    # Model definition with fixed or sampled hidden layer size
    hidden_layer = fixed_params.get("hidden_layer") or trial.suggest_int(
        "hidden_layer", search_space["hidden_layer_min"], search_space["hidden_layer_max"]
    )
    model = classes.XORSNN(hidden_layer=hidden_layer).to(config.DEVICE)
    define_initialization(trial, model, input_activity, fixed_params, search_space)

    # Learning rate and optimizer/scheduler definition
    lr = fixed_params.get("lr") or trial.suggest_float("lr", search_space["lr_min"], search_space["lr_max"], log=True)
    optimizer = define_optimizer(trial, model, lr, fixed_params, search_space)
    scheduler = define_scheduler(trial, optimizer, lr, epochs, fixed_params, search_space)
    loss_fn = torch.nn.CrossEntropyLoss()

    checkpoint_interval = study_config.get("checkpoint_interval", 10)
    run = None

    try:
        run = wandb.init(
            project="SNNTemporalXOR",
            group=study_config["study_name"],                                                   # group runs by study name
            name=f"trial-{trial.number}",
            config={"trial": trial.number, "epochs": epochs, **fixed_params, **trial.params},   # log both fixed and tuned params
            reinit="finish_previous"
        )

        checkpoint = load_checkpoint(trial, model, optimizer, scheduler, checkpoints_dir)
        start_epoch, lowest_loss = (0, float("inf")) if checkpoint is None else (checkpoint[0], checkpoint[1])

        for epoch in range(start_epoch, epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)
            avg_loss_val, acc_val = functions.evaluate(model, val_loader, loss_fn)

            if scheduler is not None:
                scheduler.step(avg_loss_val) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

            # Track lowest validation loss and save best model
            if avg_loss_val < lowest_loss:
                lowest_loss = avg_loss_val
                trial_best_dir = os.path.join(best_models_dir, f"trial_{trial.number}")
                os.makedirs(trial_best_dir, exist_ok=True)
                torch.save(
                    {
                        "trial_number": trial.number,
                        "epoch": epoch,
                        "val_loss": avg_loss_val,
                        "val_accuracy": acc_val,
                        "params": {**fixed_params, **trial.params},
                        "model_state_dict": model.state_dict()
                    },
                    os.path.join(trial_best_dir, "best_model.pt")
                )

            # Compute and log simulation timing
            current_sim_time = previous_optimization_time + (time.perf_counter() - start_time)
            study.set_user_attr("optimization_time", current_sim_time)
            hours, rem = divmod(int(current_sim_time), 3600)
            minutes, seconds = divmod(rem, 60)
            sim_time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            run.log({
                "epoch": epoch,
                "train_loss": avg_loss_train,
                "val_loss": avg_loss_val,
                "train_accuracy": acc_train,
                "val_accuracy": acc_val,
                "lr": current_lr,
                "sim_time": current_sim_time
            })
            run.summary["best_val_loss"] = lowest_loss
            run.summary["final_val_loss"] = avg_loss_val
            run.summary["final_val_accuracy"] = acc_val
            run.summary["completed_epochs"] = epoch + 1
            run.summary["final_lr"] = current_lr
            run.summary["status"] = "in progress"
            run.summary["simulation_time"] = sim_time_string

            if (epoch + 1) % checkpoint_interval == 0:
                save_checkpoint(trial, epoch, model, optimizer, scheduler, lowest_loss, checkpoints_dir)

            trial.report(avg_loss_val, step=epoch)
            if trial.should_prune():
                run.summary["status"] = "pruned"
                raise optuna.TrialPruned()

        run.summary["status"] = "completed"
    finally:
        if run is not None:
            run.finish()

    return lowest_loss

## --- MAIN CODE ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven SNN Hyperparameter Optimization")
    parser.add_argument("--config", type=str, default="test_2_nominal_50neurons.json",
                        help="Study config JSON file in optimization/inputs/ or custom path")
    args = parser.parse_args()

    study_cfg = load_study_config(args.config)                                                  # load study parameters
    study_name = study_cfg["study_name"]
    target_trials = study_cfg.get("target_trials", 1000)

    # Setup isolated study directory structure
    study_dir = os.path.join("optimization", "studies", study_name)
    checkpoints_dir = os.path.join(study_dir, "checkpoints")
    best_models_dir = os.path.join(study_dir, "best_models")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_models_dir, exist_ok=True)

    # Save a run snapshot of the configuration inside the study folder
    with open(os.path.join(study_dir, "config.json"), "w") as f:
        json.dump(study_cfg, f, indent=4)

    # Generate data loaders and compute baseline activity
    train_dataset, val_dataset = functions.generate_train_validation_dataset()
    train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)
    input_activity = compute_input_activity(train_loader)
    print(f"[{study_name}] Average input activity: {input_activity:.6f}")

    # Build sampler, pruner and persistent SQLite storage
    sampler = build_sampler(study_cfg.get("sampler", {}))
    pruner = build_pruner(study_cfg.get("pruner", {}))
    db_path = os.path.join(study_dir, f"{study_name}.db")

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )

    # Enqueue reference trials on fresh study creation
    if len(study.trials) == 0:
        for trial_params in study_cfg.get("enqueued_trials", []):
            study.enqueue_trial(trial_params)

    prev_time = study.user_attrs.get("optimization_time", 0.0)                                 # accumulated simulation time
    start_time = time.perf_counter()

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining_trials = target_trials - len(study.trials)

    print(f"[{study_name}] Recorded trials: {len(study.trials)} (completed: {len(completed_trials)}).")
    if remaining_trials <= 0:
        print(f"Target of {target_trials} trials already reached for '{study_name}'.")
    else:
        print(f"Starting {remaining_trials} remaining trials (target: {target_trials})...\n")
        study.optimize(
            lambda trial: objective(trial, study_cfg, study, train_loader, val_loader, input_activity,
                                    checkpoints_dir, best_models_dir, start_time, prev_time),
            n_trials=remaining_trials
        )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        print("\n=== STUDY COMPLETED ===")
        print("Best value:", study.best_value)
        print("Best parameters:", study.best_params)
    else:
        print("No completed trials.")