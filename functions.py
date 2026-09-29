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
import plotext as pltterm
import datetime
import os
from torch.utils.data import DataLoader

## -- INPUT FUNCTIONS ---

def XOR_no_noise(A, B, len_sign_a=config.SIGN_A, len_sign_b=config.SIGN_B, len_sign_r=config.READOUT, gap=config.GAP):
    """ Function that generates 3 input signals (Neuron A, Neuron B and Readout) without noise

    We associate to the 0 the generation of fire probability of 30% for each time step, 60% for 1

    args:
        A(int): first input to the XOR (0 or 1)
        B(int): second input to the XOR (0 or 1)
        len_sign_a(int): number of timesteps given to neuron A input
        len_sign_b(int): number of timesteps given to neuron B input
        len_sign_r(int): number of timesteps given to readout signal
        gap(int): number of timesteps between the end of signal A and the begging of signal B
    
    returns:
        Data(torch.Tensor): Matrix containing the 3 inputs randomly generates
            Shape: [3, sign_a + gap + sign_b + readout]
        label(int): result of standard XOR computation
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
    A_tot, B_tot = torch.cat((A_on, A_off)), torch.cat((B_off_b, B_on, B_off_a))

    # readout generation
    readout = torch.cat((torch.zeros(len_sign_a+gap+len_sign_b), torch.ones(len_sign_r)))

    return torch.stack((A_tot, B_tot, readout), dim = 0), A ^ B

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


## --- NN functions ---

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
    acc_train = (total_correct / total_samples) * 100                       # accuracy in this epoch

    return avg_loss_train, acc_train

@torch.no_grad()
def test(model, test_loader, loss_fn):
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
    acc_test = (total_correct / total_samples) * 100
    return avg_loss_test, acc_test

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
    current_date = datetime.datetime.now().strftime("%Y%m%d")                               # get the current date
    modelname = f"models/{current_date}-E{config.EPOCHS}-LR{config.LEARNING_RATE}.pth"      # decide the model name
    torch.save(model.state_dict(), modelname)
    print("model correctly saved in 'models' folder\n")

def load_model():
    """ 
    Function that proceeds to create the string and load the corresponding 
    model based on the parameters stored in "config.py" before the compiling

    Args:
        None

    Returns:
        model (torch.nn.Module): The neural network model trained in a previous session,
                                 or None if not found
    """

    check_string = f"E{config.EPOCHS}-LR{config.LEARNING_RATE}"                 # create the model we have to look for
    folder_name = "models"                                                       # directory
    if os.path.exists(folder_name):
        for filename in os.listdir(folder_name):
            if filename.endswith(".pth") and check_string in filename:          # ensure to choose correct name and that is model
                file_path = os.path.join(folder_name, filename)                 # creation of complete path
                model = classes.XORSNN()
                weights = torch.load(file_path)                                 # load model
                model.load_state_dict(weights)
                print(f"Model correctly loaded from: {file_path}")
                return model
    print(f"No model found with the parameters: {check_string}")                # Handle the case with no model present with that name
    return None

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
                no_date = filename[10:-4]           # YYYYMMDD-eVAL1-lrVAL2.pth -> VAL1-e-VAL2
                split_dashed = no_date.split("-")   # VAL1-e-VAL2 -> ["VAL1", "e", "VAL2"]
                epochs = int(split_dashed[0])
                lr = float(split_dashed[1][2:])
                if lr == config.LEARNING_RATE and epochs == config.EPOCHS:
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
    acc_list = []
    for _ in range(config.N_ROBUSTNESS_DATASET):
        current_test_loader = generate_new_random_data()
        loss, acc = test(model, current_test_loader, loss_fn)
        acc_list.append(acc)
    print(f"The accuracy we reached is:")
    print(f"    -> worst accuracy: {min(acc_list):.2f}%")
    print(f"    -> Best accuracy: {max(acc_list):.2f}%")
    print(f"    -> Average accuracy: {sum(acc_list)/len(acc_list):.2f}%")

## --- Plotting functions ---

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

    plt.clf()                           # fundamental to avoid tracking 2 graphs superimposed if you call the function twice
    pltterm.subplots(3, 1)              # generation of a plot with 3 subplots
    pltterm.plotsize(100, 30)           # increase graph dimensions to be easily seen in the terminal

    for i in range(3):
        pltterm.subplot(i+1, 1)                                                # here the index starts from 1!
        spike_positions = torch.nonzero(Data[i, :] == 1).squeeze().numpy()     # array with spike positions
        pltterm.eventplot(spike_positions)       # draws the spikes in the position of spike_positions
        pltterm.ylabel("Spikes")
        pltterm.title("Segnale: "+channels[i])
        pltterm.xlim(-1, Data.shape[1])

    pltterm.subplot(3,1)                # consider last graph to put a unique label for the 3 x axes
    pltterm.xlabel("Time")
    pltterm.show()
    pltterm.clear_figure()              