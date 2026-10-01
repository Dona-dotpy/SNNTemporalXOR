"""
    Project that aims to find the best parameters for a chosen SNN.
    The idea is to pass in brute force every or some of the possible schedulers / initilization / loss functions ... and to find the optimal parameters that 
    make the net work on that specific task
"""

## --- LIBRARIES --- 

import optuna
import wandb

import torch
import classes, functions, config

## --- FUNCTIONS ---

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

    run = wandb.init(                                                                               # initialize weights and biases to see training steps on iPhone
                project="SNNTemporalXOR",
                name=f"trial-{trial.number}",
                config={
                    "lr": lr,
                    "epochs": config.EPOCHS
                },
                reinit="finish_previous"
            )

    # use try/finally structure to ensure closure of the wandb communication in case of any error
    try:        
        # parameters of the net and optimizators
        model = classes.XORSNN().to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        lowest_loss = float("inf")                                                                      # generate +inf since we are in front of a minimization problem
        for epoch in range(config.EPOCHS):                                                              # train loop
            avg_loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)
            avg_loss_val, acc_val = functions.evaluate(model, val_loader, loss_fn)

            run.log({                                                                                 # send data to iPhone
                "epoch": epoch,
                "train_loss": avg_loss_train,
                "val_loss": avg_loss_val,
                "train_accuracy": acc_train,
                "val_accuracy": acc_val,
                "lr": optimizer.param_groups[0]["lr"]
            })          

            if avg_loss_val < lowest_loss:                                                              # keep the lowest validation loss reached during training, not only the final one
                lowest_loss = avg_loss_val

            trial.report(avg_loss_val, step=epoch)                                                      # reports the current validation loss to Optuna for this epoch  

            if trial.should_prune():                                                                    # check if it's convenient an early stop
                raise optuna.TrialPruned()

    finally:
        run.finish()            # interrupt iPhone transmission
    
    return lowest_loss

## --- MAIN CODE ---

train_dataset, val_dataset = functions.generate_train_validation_dataset()
train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)

sampler = optuna.samplers.TPESampler(seed = config.SEED)                        # decides which values oh hyperparameters we have to try in the next trial, in this case it's based on best results of previus trials
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)    # for a minimization study, prunes trials whose intermediate loss is worse than the median of previous completed trials (1st 5 trials completed interely, never stop before epoch 100)

study = optuna.create_study(                                        # creates a new optimization study aimed at minimizing the objective value and stores it in a Study object that manages the trials
    study_name = "learning_rate_optimization",
    direction="minimize",
    sampler = sampler,
    pruner = pruner,
    storage="sqlite:///optimization_results/learning_rate/learning_rate_optimization.db",
    load_if_exists=True
    )                   
study.optimize(                                                     # we pass to the study object what we must minimize and in how many trials
    lambda trial: objective(trial, train_loader, val_loader),       # Optuna expects a function callable as f(trial); we map objective(trial, train_ld, test_ld) -> f(trial)       
    n_trials=30,    
    )              

print("Best value:", study.best_value)
print("Best parameters:", study.best_params)


