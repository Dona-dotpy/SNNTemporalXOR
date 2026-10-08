"""
Main Execution and Interactive Pipeline for Temporal XOR SNN.
Orchestrates the standard workflow:
1. Displays environment info, active device, and experimental parameters.
2. Generates rate-coded datasets and previews spike trains in the terminal.
3. Builds the LIF network and applies fluctuation-driven weight initialization.
4. Trains the model (with best-model checkpointing) or interactively loads 
   an existing checkpoint from 'models/' to run immediate evaluation.
"""

## --- Libraries ---
import classes, functions, config

import torch
import torch.nn as nn

## --- MAIN CODE ---

# check of the device
functions.print_parameters()

# Generation of Dataset
train_dataset, test_dataset = functions.generate_train_validation_dataset()
# generation of DataLoader
train_loader, test_loader = functions.generate_train_validation_dataloader(train_dataset, test_dataset)

# first try in the generation of the temporal XOR   
print(f"Let's see which type of input we have:\n")
Data, _ = next(iter(train_loader))
functions.plot_inputs_in_terminal(Data[0, :, :])

# model, loss function and optimizer definition
model = classes.XORSNN().to(config.DEVICE)
loss_fn = config.LOSS_FN()
functions.fluctuation_driven_initialization(model, train_loader)                        # Initialization
optimizer = config.OPTIMIZER( model.parameters(), **config.OPTIMIZER_PARAMS)
scheduler = config.SCHEDULER( optimizer, **config.SCHEDULER_PARAMS)                     # Scheduler

# check if the model with these parameters has already been trained, skip training otherwise
if not functions.check_trained():
    functions.train_and_validate(model, train_loader, test_loader, optimizer, loss_fn, scheduler)
else:
    print(f"\nModel with epochs = {config.EPOCHS} and learning rate = {config.LEARNING_RATE} has already been trained")

    # Creation of input choice to decide whether we have to train again or not
    while True:
        choice = input(f"[y/n]  Do you want to train it again?  ").strip().lower()              # get keyboard IN, remove void spaces and make everything lowercase
        if choice == "y":
            functions.train_and_validate(model, train_loader, test_loader, optimizer, loss_fn, scheduler)
            break
        elif choice == "n":
            # Choice regarding which model we want to load
            print("Are you looking for a specific model?")
            while True:
                choice = input("[y/n]   Say 'y' if you are looking for a specific model, 'n' if I can load deafult model:  ").strip().lower()
                if choice == "y":
                    epochs = input("Which is the number of training epochs?  ").strip()
                    lr = input("Which is the learning rate?  ").strip()
                    hidden = input("How many neurons there are per hidden layer?   ").strip()
                    beta = input("Which is the Beta parameter of the neuron?   ").strip()
                    print("Let's load model parameters and test model robustness... ")
                    model = functions.load_model(epochs, lr, hidden, beta)
                    break
                elif choice == "n":
                    print("Let's load default model and the the robustness...")
                    model = functions.load_model()
                    break
                else:
                    print("Input not recognized. Please insert 'y' for yes and 'n' for no")
            functions.test_model_robustness(model, loss_fn)
            break
        else:
            print("Input not recognized. Please insert 'y' for yes and 'n' for no")

