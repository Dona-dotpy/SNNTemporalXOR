# Dataset generation of to train the net for temporal XOR

## --- Libraries ---
import classes, functions, config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import plotext as plt


## --- MAIN CODE ---

# check of the device
print(
    "\n============ DEVICE ================\n"
    f"==      We are working on {config.DEVICE}     ==\n"
    "====================================\n"
    )

# Generation of Dataset
torch.manual_seed(142)                                                   # Seed setup
dataset = classes.TemporalXORDataset()                                  # Dataset allocation
n_train = int(config.TRAIN_SPLIT * len(dataset))                        # length of train dataset
n_test = len(dataset) - n_train                                         # length of test dataset
train_dataset, test_dataset = random_split(                                                       
    dataset,
    [n_train, n_test],
    generator = torch.Generator().manual_seed(42)
)

# generation of DataLoader
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print("Train dataset e Test dataset successfully generated.")
print(f"The length of the train dataset is: {len(train_dataset)}")
print(f"The length of the test dataset is: {len(test_dataset)}\n")

# first try in the generation of the temporal XOR   
print(f"Let's see which type of input we have:\n")
Data, _ = next(iter(train_loader))
functions.plot_inputs_in_terminal(Data[0, :, :])

# model, loss function and optimizer definition
model = classes.XORSNN().to(config.DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

# check if the model with these parameters has already been trained, skip training otherwise
if not functions.check_trained():
    print("\nTraining underway ...\n")
    loss_hist_train, acc_hist_train = [], []
    loss_hist_test, acc_hist_test = [], []

    best_accuracy_so_far  = 0.0
    for epoch in range(config.EPOCHS):
        avg_loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)
        avg_loss_test, acc_test = functions.test(model, test_loader, loss_fn)
        loss_hist_train.append(avg_loss_train), acc_hist_train.append(acc_train)
        loss_hist_test.append(avg_loss_test), acc_hist_test.append(acc_test)

        # if we get a good accuracy, store the model (this avoid to save the model only at the end of the last epoch
        # in order to avoid saving parameters if the training is making the model worse)
        if acc_test > best_accuracy_so_far:
            best_accuracy_so_far = acc_test
            functions.store_model(model)

    print(f"Training completed on {config.EPOCHS} epochs")
    print(f"The best accuracy we reached in training is: {max(acc_hist_train)}%")
    print(f"The best accuracy we reached in testing is: {max(acc_hist_test)}%\n")
else:
    print(f"Model with epochs = {config.EPOCHS} and learning rate = {config.LEARNING_RATE} has already been trained")
    print("Let's load model parameters ... ")
    model = functions.load_model()
    functions.test_model_robustness(model, loss_fn)







