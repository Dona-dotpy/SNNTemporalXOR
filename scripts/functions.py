"""
Utilities for the Temporal XOR project.

This module contains all the necessary helper functions for data generation, 
training, and evaluation of the Spiking Neural Network (SNN). It also provides 
tools for visualizing results and membrane potentials.

Key features:
    * Data Generation: Creation of time-series sequences (Inputs A, B, and Readout).
    * Training & Validation: Training loops, loss computation, and validation steps.
    * Plotting: Functions to visualize spikes, weights, and membrane potentials.
"""

import config, classes

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotext as pltterm
import datetime
import os
from torch.utils.data import DataLoader, random_split
import time

## -- INPUT FUNCTIONS AND DATA---

def XOR(A, B, len_sign_a=config.SIGN_A, len_sign_b=config.SIGN_B, len_sign_r=config.READOUT, gap=config.GAP, noise_spikes = config.NOISE_SPIKES):
    """ Function that generates 3 input signals (Neuron A, Neuron B and Readout)

    We associate to the 0 the generation of fire probability of 30% for each time step, 60% for 1

    args:
        A(int): first input to the XOR (0 or 1)
        B(int): second input to the XOR (0 or 1)
        len_sign_a(int): number of timesteps given to neuron A input
        len_sign_b(int): number of timesteps given to neuron B input
        len_sign_r(int): number of timesteps given to readout signal
        gap(int): number of timesteps between the end of signal A and the begging of signal B
        noise_spikes(int): number of disturbance spikes injected into the temporal gap
    
    returns:
        Data(torch.Tensor): Matrix containing the 3 inputs randomly generates
            Shape: [3, sign_a + gap + sign_b + readout]
        label(int): result of standard XOR computation
        noise_metrics(float): normalized centre of mass of the noise spikes inside the gap [0,1]
    """

    A_on, B_on = torch.zeros(len_sign_a), torch.zeros(len_sign_b)

    # Let's generate random permutations and take 1st 3 or 6 numbers to decide spike positions
    A_on, B_on = torch.zeros(len_sign_a), torch.zeros(len_sign_b)
    spikes_0_A, spikes_1_A = int(len_sign_a*config.PROB_0), int(len_sign_a*config.PROB_1)
    spikes_0_B, spikes_1_B = int(len_sign_b*config.PROB_0), int(len_sign_b*config.PROB_1)
    if A:
        A_on_indexes = torch.randperm(len_sign_a)[:spikes_1_A]
    else:
        A_on_indexes = torch.randperm(len_sign_a)[:spikes_0_A]
    if B:
        B_on_indexes = torch.randperm(len_sign_b)[:spikes_1_B]
    else:
        B_on_indexes = torch.randperm(len_sign_b)[:spikes_0_B]

    # let's generate spikes in the positions we randomly generated
    A_on[A_on_indexes] = 1
    B_on[B_on_indexes] = 1

    # Let's complete the tensor with the off status
    A_off = torch.zeros(gap + len_sign_b + len_sign_r)
    B_off_b, B_off_a = torch.zeros(len_sign_a + gap), torch.zeros(len_sign_r)

    # noise generation
    noise_positions = torch.randperm(2*gap)[:noise_spikes]          # consider both channels
    for i in noise_positions:                                       # depending on the value put the spike on channel A or B
        if i >= gap:
            B_off_b[len_sign_a+i-gap] = 1
        else:
            A_off[i] = 1

    # compute noise metrics (sum of indexes of spikes, considering that if on B we have to remove gap)
    noise_metrics = (sum(noise_positions) - gap * (noise_positions>=gap).float().sum())/ (gap*len(noise_positions)) if gap > 0 and len(noise_positions) >0 else 0

    # create total tensor
    A_tot, B_tot = torch.cat((A_on, A_off)), torch.cat((B_off_b, B_on, B_off_a))

    # readout generation
    readout = torch.cat((torch.zeros(len_sign_a+gap+len_sign_b), torch.ones(len_sign_r)))

    return torch.stack((A_tot, B_tot, readout), dim = 0), A ^ B, noise_metrics

def generate_new_random_data():
    """
    function that creates a new dataLoader different form the ones for training and
    testing in order to check the robustness a model we are loading.

    Args:
        dataset (classes.temporalXORDataset): The dataset we want to use to build the dataloader

    Returns:
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the new test set.
    """
    return DataLoader(classes.TemporalXORDataset(), batch_size=config.BATCH_SIZE, shuffle=False)

def generate_train_validation_dataset(len_sign_a=config.SIGN_A, len_sign_b=config.SIGN_B, len_sign_r=config.READOUT, 
                                      gap=config.GAP, noise_spikes = config.NOISE_SPIKES):
    """
    Function that generates the Temporal XOR dataset and splits it into a train
    and a test set, using fixed seeds for reproducibility.

    Args:
        len_sign_a(int): number of temporal spikes allocated to input A
        len_sign_b(int): number of temporal spikes allocated to input B
        len_sign_r(int): number of temporal spikes allocated to readout
        gap(int): number of temporal spikes allocated to gap
        noise_spikes(int): number of disturbance spikes injected into the temporal gap


    Returns:
        train_dataset (torch.utils.data.Subset): The subset used for training.
        val_dataset (torch.utils.data.Subset): The subset used for testing.
    """
    torch.manual_seed(config.SEED)                                                      # Seed setup
    dataset = classes.TemporalXORDataset(len_sign_a=len_sign_a, len_sign_b=len_sign_b,  # Dataset allocation
                                len_sign_r=len_sign_r, gap=gap, noise_spikes=noise_spikes)   
    n_train = int(config.TRAIN_SPLIT * len(dataset))                                    # length of train dataset
    n_val = len(dataset) - n_train                                                     # length of test dataset
    train_dataset, val_dataset = random_split(                                                       
        dataset,
        [n_train, n_val],
        generator = torch.Generator().manual_seed(config.SEED)
    )

    print("Train dataset e Validation dataset successfully generated.")
    print(f"The length of the train dataset is: {len(train_dataset)}")
    print(f"The length of the test dataset is: {len(val_dataset)}\n")
    return train_dataset, val_dataset

def generate_test_dataset(len_sign_a=config.SIGN_A, len_sign_b=config.SIGN_B, len_sign_r=config.READOUT, 
                          gap=config.GAP, noise_spikes = config.NOISE_SPIKES, seed = config.SEED):
    """
    Function that generates the Temporal XOR dataset  for testing, using fixed seeds 
    for reproducibility.

    Returns:
        test_dataset (torch.utils.data.DataSet): dataset used for testing.
    """   
    torch.manual_seed(seed)
    dataset = classes.TemporalXORDataset(len_sign_a=len_sign_a, len_sign_b=len_sign_b,          # Dataset allocation
                                    len_sign_r=len_sign_r, gap=gap, noise_spikes=noise_spikes)
    print("Test dataset successfully generated.")
    print(f"The length of the test dataset is: {len(dataset)}")
    return dataset

def generate_train_validation_dataloader(train_dataset, val_dataset):
    """ Generate train and test dataloaders from datasets

    Args:
        train_dataset (torch.utils.data.Subset): The subset used for training.
        val_dataset (torch.utils.data.Subset): The subset used for testing.
    
    Returns:
        train_dataloader (torch.utils.data.DataLoader): The DataLoader for the train set (shuffled).
        val_dataloader (torch.utils.data.DataLoader): The DataLoader for the test set (not shuffled).
    """
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_dataloader, val_dataloader

def generate_test_dataloader(test_dataset):
    """ Generate test dataloaders from datasets

    Args:
        test_dataset (torch.utils.data.Subset): The dataset used for testing.
    
    Returns:
        test_dataloader (torch.utils.data.DataLoader): The DataLoader for the test set (not shuffled).
    """   
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return test_dataloader

def fluctuation_driven_initialization(model, train_loader):
    """
    Applies fluctuation-driven initialization to the Linear layers
    of the SNN using the average activity of the training dataset.

    Args:
        model (torch.nn.Module): model to initialize.
        train_loader (torch.utils.data.DataLoader): training dataloader.

    Returns:
        None
    """

    total_activity = 0.0
    total_elements = 0

    for data, _ in train_loader:
        total_activity += data.sum().item()
        total_elements += data.numel()

    input_activity = total_activity / total_elements

    for i, module in enumerate(model.net):
        if isinstance(module, torch.nn.Linear):
            lif_layer = model.net[i + 1]
            threshold = lif_layer.threshold.item()
            beta = config.BETA
            fan_in = module.in_features
            sigma_u = threshold / config.FLUCT_XI
            epsilon_hat = 1.0 / (1.0 - beta ** 2)
            weight_std = sigma_u / (fan_in * input_activity * epsilon_hat) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=weight_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

## --- TRAIN AND TEST FUNCTIONS ---

def train(model, train_loader, optimizer, loss_fn):
    """ Executes the training loop for the Spiking Neural Network over one epoch.

    The process relies on rate coding: output spikes are summed along the 
    temporal dimension to obtain firing rates (logits), which are then passed 
    to the loss function. Computes gradients and updates weights via backpropagation.

    Args:
        model (torch.nn.Module): The SNN model to be trained.
        train_loader (torch.utils.data.DataLoader): Iterable dataset for training.
        loss_fn (callable): Function to compute the error (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for weight updates (e.g., Adam).

    Returns:
        tuple: A tuple containing:
            - avg_loss_train (float): Average loss computed over the entire training set.
            - best_acc_train (float): Maximum accuracy (in percentage) recorded across training batches.
    """

    loss_train_hist = []

    total_correct, total_samples = 0, 0

    model.train()
    for data, label in train_loader:
        data, label = data.to(config.DEVICE), label.to(config.DEVICE)       # put data on GPU if present

        spk_out, mem_out = model(data)                                      # do the forward pass
        total_spikes = spk_out.sum(dim=2)                                   # let's compute how many times each neuron fired at point of the batch
        loss_train = loss_fn(total_spikes, label)                           # compute the loss -ln(e^num_cor / (e^num_cor + e^num_wrong))
        optimizer.zero_grad()                                               # zero the gradient
        loss_train.backward()                                               # compute gradients (d Loss/d weight) for all weights via backpropagation
        optimizer.step()                                                    # update weights: w_new = w - lr * (m / (sqrt(v) + eps))

        pred = torch.argmax(total_spikes, dim=1)                            # compute which is the predicted output from the number of spikes
        total_correct += (pred == label).float().sum()                      # count the number of correct predictions
        total_samples += label.size(0)                                      # total number of predictions
        
        loss_train_hist.append(loss_train.item())                           # store results

    avg_loss_train = sum(loss_train_hist) / len(train_loader)               # average loss in this epoch
    acc_train = ((total_correct / total_samples) * 100).item()              # accuracy in this epoch (.item() to convert a tensor into a float)

    return avg_loss_train, acc_train

@torch.no_grad()
def evaluate(model, test_loader, loss_fn):
    """ Executes the evaluation loop for the Spiking Neural Network on the test dataset.

    The network is set to evaluation mode and gradient tracking is disabled to 
    optimize memory and computation. Spikes are summed over the temporal dimension 
    to obtain firing rates (logits), which are then evaluated against the target labels.

    Args:
        model (torch.nn.Module): The SNN model to be evaluated.
        test_loader (torch.utils.data.DataLoader): Iterable dataset for testing.
        loss_fn (callable): Function to compute the error (e.g., CrossEntropyLoss).

    Returns:
        tuple: A tuple containing:
            - avg_loss_test (float): Average loss computed over the entire test set.
            - acc_test (float): Accuracy of the Dataset.
    """

    loss_test_hist = []
    total_correct, total_samples = 0,0 

    model.eval()
    for data, label in test_loader:
        data, label = data.to(config.DEVICE), label.to(config.DEVICE)       # put data on GPU if present
        
        spk_out, mem_out = model(data)                                      # do the forward pass
        total_spikes = spk_out.sum(dim=2)                                   # let's compute how many times each neuron fired at point of the batch
        loss_test = loss_fn(total_spikes, label)                            # compute the loss -ln(e^num_cor / (e^num_cor + e^num_wrong))
        pred = torch.argmax(total_spikes, dim=1)                            # compute which is the predicted output from the number of spikes
        total_correct += (pred == label).float().sum()                      # count the number of correct predictions
        total_samples += label.size(0)                                      # total number of predictions

        loss_test_hist.append(loss_test.item())                             # store results


    avg_loss_test = sum(loss_test_hist) / len(test_loader)
    acc_test = ((total_correct / total_samples) * 100).item()
    return avg_loss_test, acc_test

def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, scheduler):
    """
    Executes the complete training and evaluation pipeline for the Temporal XOR SNN.

    This function iterates through the number of epochs defined in the configuration,
    alternating between training the model on the rate-coded training set and evaluating 
    it on the test set. It implements a best-model selection strategy by monitoring the 
    validation loss, automatically saving the model weights to disk whenever a new 
    performance peak is reached to prevent storing overfitted parameters.

    Args:
        model (torch.nn.Module): The Spiking Neural Network to be trained and evaluated.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the rate-coded training samples.
        test_loader (torch.utils.data.DataLoader): DataLoader providing the rate-coded testing samples.
        optimizer (torch.optim.Optimizer): The optimization algorithm used to update the network's weights.
        loss_fn (callable): The criterion used to compute the error gradient.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning-rate scheduler updated once per epoch.

    Returns:
        None. The model is modified in-place during optimization, and the optimal weights 
        are saved directly to the disk via the `store_model()` function.
    """

    print("\nTraining underway ...\n")
    loss_hist_train, acc_hist_train = [], []
    loss_hist_test, acc_hist_test = [], []

    best_loss_so_far  = float("inf")
    start_time = time.perf_counter()

    for _ in range(config.EPOCHS):
        avg_loss_train, acc_train = train(model, train_loader, optimizer, loss_fn)
        avg_loss_val, acc_val = evaluate(model, val_loader, loss_fn)
        loss_hist_train.append(avg_loss_train), acc_hist_train.append(acc_train)
        loss_hist_test.append(avg_loss_val), acc_hist_test.append(acc_val)

        scheduler.step()

        # if we get a good loss, store the model (this avoid to save the model only at the end of the last epoch
        # in order to avoid saving parameters if the training is making the model worse)
        if avg_loss_val < best_loss_so_far:
            best_loss_so_far = avg_loss_val
            store_model(model)

    tot_time = time.perf_counter() - start_time
    minutes, seconds = int(tot_time // 60), tot_time % 60 
        
    print(f"Training completed on {config.EPOCHS} epochs")
    print(f"The best accuracy we reached in training is: {max(acc_hist_train)}%")
    print(f"The best accuracy we reached in validation is: {max(acc_hist_test)}%\n")
    print(f"Test time [min:sec]: {minutes}:{seconds:05.2f}\n")
    plot_loss_accuracy(loss_hist_train, loss_hist_test, acc_hist_train, acc_hist_test)

def check_trained():
    """ 
    Based on the parameters LEARNING rate and EPOCHS, looks if a model with the exact same parameters
    has already been trained in the past

    Args:
        None (parameters are not variables)

    returns:
        bool
    """
    folder_name = "models"

    # check existance of the folder
    if os.path.exists(folder_name):
        # iters on the model names inside the folder to look for learing rate and epochs
        for filename in os.listdir(folder_name):
            if filename.endswith(".pth"):           # avoids to analyse file automatically generated from operative system
                split_dashed = filename.split("-")   # EVAL1-LRVAL2-HVAl3-BVAL4 -> [EVAL1, LRVAL2, HVAL3, BVAL4.pth]
                epochs = int(split_dashed[0][1:])
                lr = float(split_dashed[1][2:])
                hidden = int(split_dashed[2][1:])
                beta = float(split_dashed[3][1:-4])

                if lr == config.LEARNING_RATE and epochs == config.EPOCHS and hidden == config.HIDDEN_LAYER and beta == config.BETA:
                    return True
    return False

def test_model_robustness(model, loss_fn):
    """
    Functions to evaluate the robustness of a model previously trained by generating 
    a new dataset different from the one create in "main.py" to test and train a new model

    Args:
        model (classes.XORSNN): the model we have to test
        loss_fn (nn.CrossEntropyLoss usually): the loss function we want to use
    
    Returns:
        None
    """
    loss_list, acc_list = [], []
    start_time = time.perf_counter()
    for _ in range(config.N_ROBUSTNESS_DATASET):
        current_test_loader = generate_new_random_data()
        loss, acc = evaluate(model, current_test_loader, loss_fn)
        acc_list.append(acc)
        loss_list.append(loss)
    print(f"The accuracy we reached is:")
    print(f"    -> worst accuracy: {min(acc_list):.2f}%")
    print(f"    -> Best accuracy: {max(acc_list):.2f}%")
    print(f"    -> Average accuracy: {sum(acc_list)/len(acc_list):.2f}%\n")

    tot_time = time.perf_counter() - start_time
    minutes, seconds = int(tot_time // 60), tot_time % 60 
    print(f"Test time [min:sec]: {minutes}:{seconds:05.2f}\n")
    print("Let's see the loss and accuracy trends ...\n")
    plot_loss_accuracy_in_terminal(loss_test=loss_list, acc_test=acc_list)


## --- SAVE FUNCTIONS ---

def store_model(model):
    """Saves the PyTorch model weights with a formatted timestamp.

    Checks if the 'models' directory exists, creates it if necessary, 
    and saves the model's state dictionary using the format: 
    YYYYMMDD-E{EPOCHS}-LR{LR}.pth.

    Args:
        model (torch.nn.Module): The trained neural network model.

    Returns:
        None
    """
    os.makedirs("models", exist_ok=True)                                                    # create model folder if not present
    modelname = f"models/E{config.EPOCHS}-LR{config.LEARNING_RATE}-H{config.HIDDEN_LAYER}-B{config.BETA}.pth"      # decide the model name
    torch.save(model.state_dict(), modelname)

def load_model(epochs = config.EPOCHS, lr = config.LEARNING_RATE, hidden = config.HIDDEN_LAYER, beta=config.BETA):
    """ 
    Function that proceeds to create the string and load the corresponding 
    model based on the parameters stored in "config.py" before the compiling

    Args:
        None

    Returns:
        model (torch.nn.Module): The neural network model trained in a previous session,
                                 or None if not found
    """

    check_string = f"E{epochs}-LR{lr}-H{hidden}-B{beta}"                        # create the model we have to look for
    folder_name = "models"                                                      # directory
    if os.path.exists(folder_name):
        for filename in os.listdir(folder_name):
            if filename.endswith(".pth") and check_string in filename:          # ensure to choose correct name and that is model
                file_path = os.path.join(folder_name, filename)                 # creation of complete path
                model = classes.XORSNN(hidden_layer=int(hidden), beta=float(beta))
                weights = torch.load(file_path, map_location = config.DEVICE)   # load weights on the working device
                model.load_state_dict(weights)
                print(f"Model correctly loaded from: {file_path}")
                return model.to(config.DEVICE)
    print(f"No model found with the parameters: {check_string}")                # Handle the case with no model present with that name
    return None

## --- PLOTTING AND PRINTING FUNCTIONS ---

def print_parameters():
    """ Function to print some relevant parameters inside config.py
    """
    print("\n===========================================================================")
    print(f"|    Working device:     {config.DEVICE}                                              |")
    print("|                                                                         |")
    print("|    Net parameters are the following:                                    |")
    print(f"|        Structure of the Spiking neural network:    {config.INPUT_LAYER} -> {config.HIDDEN_LAYER} -> {config.OUTPUT_LAYER}         |")
    print(f"|        Membrane potential decay factor (β):    {config.BETA}                     |")
    print("|                                                                         |")
    print("|    Input properties:                                                    |")
    print(f"|       Time setup (Signal A, Gap, Signal B, Readout):    {config.SIGN_A}->{config.GAP}->{config.SIGN_B}->{config.READOUT}  |")
    print(f"|       Encoding: 0 -> {int(config.PROB_0 * 10)} spikes, 1 -> {int(config.PROB_1 * 10)} spikes                            |")
    print("|                                                                         |")
    print("|    Training properties:                                                 |")
    print(f"|       Learning rate: {config.LEARNING_RATE:.4f}                                             |")
    print(f"|       Number of epochs: {config.EPOCHS}                                            |")
    print("|                                                                         |")
    print("===========================================================================\n")

def plot_inputs(Data: torch.Tensor):
    """ Plots the 3 inputs showing each spike

    Args:
        Data: torch.Tensor([BATCH_SIZE, FEATURES, TIMESTEPS])

    Return:
        None
    """

    if Data.dim() != 2:
            raise ValueError(f"function expects a tensor of size 2, you gave {Data.dim}")

    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharey=True)
    channels = ['A', 'B', 'R']
    for i in range(3):
        spike_positions = torch.nonzero(Data[i, :] == 1).squeeze().numpy()     # array with spike positions
        ax[i].eventplot(spike_positions, linewidths=3.0, colors='black')       # draws the spikes in the position of spike_positions
        ax[i].set_ylabel("Spikes")
        ax[i].set_title("Segnale: "+channels[i])
        ax[i].set_xlim(-1, Data.shape[1])

    fig.suptitle("Input in time", fontsize = 14)
    ax[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

def plot_inputs_in_terminal(Data: torch.Tensor):
    """ Plot of the 3 inputs showing each spike in terminal


    Args:
        Data: torch.Tensor([FEATURES, TIMESTEPS])
    
    Returns:
        None
    """
    if Data.dim() != 2:
        raise ValueError(f"function expects a tensor of size 2, you gave {Data.dim()}")

    channels = ['A', 'B', 'R']          # names of the channels

    pltterm.clf()                       # pltterm (not plt): avoids superimposing 2 graphs if you call the function twice
    pltterm.subplots(3, 1)              # generation of a plot with 3 subplots
    pltterm.plotsize(100, 30)           # increase graph dimensions to be easily seen in the terminal

    for i in range(3):
        pltterm.subplot(i+1, 1)                                                # here the index starts from 1!
        spike_positions = torch.nonzero(Data[i, :] == 1).flatten().numpy()     # array with spike positions
        pltterm.eventplot(spike_positions)                                     
        pltterm.ylim(0, 2)                                                     # spikes drawn at half height
        pltterm.yticks([])                                                     # y axis has no meaning here
        pltterm.ylabel("Spikes")
        pltterm.title("Segnale: " + channels[i])
        pltterm.xlim(-1, Data.shape[1])
        pltterm.xticks(list(range(0, Data.shape[1] + 1, 10)))

    pltterm.subplot(3, 1)               # consider last graph to put a unique label for the 3 x axes
    pltterm.xlabel("Time")
    pltterm.show()
    pltterm.clear_figure()

def plot_loss_accuracy(loss_train = None, loss_test = None, acc_train = None, acc_test = None):
    """
    Function that plots loss and accuracy through epochs.

    Train and test are plotted on separate rows (loss on the left, accuracy on the
    right). A row is drawn only if both its loss and accuracy histories are given.

    Args:
        loss_train (list[float]): Train loss for each epoch (optional).
        loss_test (list[float]): Test loss for each epoch (optional).
        acc_train (list[float]): Train accuracy (%) for each epoch (optional).
        acc_test (list[float]): Test accuracy (%) for each epoch (optional).

    Returns:
        None
    """
    rows = [(name, loss, acc)
            for name, loss, acc in [("train", loss_train, acc_train), ("test", loss_test, acc_test)]
            if loss is not None and acc is not None]
    if not rows:
        raise ValueError("Provide at least one pair (loss, accuracy)")

    fig, ax = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)), squeeze=False)
    for i, (name, loss, acc) in enumerate(rows):
        epochs = range(1, len(loss) + 1)
        ax[i, 0].plot(epochs, loss)
        ax[i, 0].set_title(f"Loss {name}")
        ax[i, 0].set_ylabel("Loss")
        ax[i, 1].plot(epochs, acc)
        ax[i, 1].set_title(f"Accuracy {name}")
        ax[i, 1].set_ylabel("Accuracy [%]")
        for a in ax[i]:
            a.set_xlabel("Epoch")

    plt.tight_layout()
    plt.show()

def plot_loss_accuracy_in_terminal(loss_train=None, loss_test=None, acc_train=None, acc_test=None):
    """
    Function that plots loss and accuracy through epochs in the terminal.

    Train and test are plotted on separate rows (loss on the left, accuracy on the
    right). A row is drawn only if both its loss and accuracy histories are given.

    Args:
        loss_train (list[float]): Train loss for each epoch (optional).
        loss_test (list[float]): Test loss for each epoch (optional).
        acc_train (list[float]): Train accuracy (%) for each epoch (optional).
        acc_test (list[float]): Test accuracy (%) for each epoch (optional).

    Returns:
        None
    """
    rows = [(name, loss, acc)
            for name, loss, acc in [("train", loss_train, acc_train), ("test", loss_test, acc_test)]
            if loss is not None and acc is not None]
    if not rows:
        raise ValueError("Provide at least one pair (loss, accuracy)")

    pltterm.clf()                                   # avoids superimposing graphs if the function is called twice
    pltterm.subplots(len(rows), 2)
    pltterm.plotsize(120, 20 * len(rows))

    for i, (name, loss, acc) in enumerate(rows):
        epochs = list(range(1, len(loss) + 1))

        pltterm.subplot(i + 1, 1)                   # here the index starts from 1!
        pltterm.plot(epochs, loss)
        pltterm.title(f"Loss {name}")
        pltterm.xlabel("Epoch")
        pltterm.ylabel("Loss")

        pltterm.subplot(i + 1, 2)
        pltterm.plot(epochs, acc)
        pltterm.title(f"Accuracy {name}")
        pltterm.xlabel("Epoch")
        pltterm.ylabel("Accuracy [%]")

    pltterm.show()
    pltterm.clear_figure()

