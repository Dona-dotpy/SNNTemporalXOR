"""
The aim of this script is to test the robustness of the script under different conditions
"""

## --- Libraries ---

import functions, config, classes

import torch
import matplotlib.pyplot as plt
import json
import os 
import csv
from datetime import datetime

## --- Constants ---

SEEDS = [i for i in range(1000,1020)]
GAPS = [0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]
## --- Functions ---

def robustness_test_different_seeds(model_class, train_loader, val_loader, optimizer_class=config.OPTIMIZER,
                                    optimizer_params = config.OPTIMIZER_PARAMS, loss_fn_class=config.LOSS_FN, 
                                    scheduler_class = config.SCHEDULER, scheduler_params = config.SCHEDULER_PARAMS):

    """
    Tests the robustness of the training procedure across different random seeds.

    For each seed defined in `SEEDS`, a new model is instantiated and trained
    from scratch using the same training and validation datasets and the same
    hyperparameter configuration.

    Changing the random seed generates a different realization of the stochastic
    components of the training procedure, including the initial model parameters
    and, when applicable, the order in which training samples are presented.

    The purpose of this test is to verify whether the hyperparameter configuration
    selected during hyperparameter optimization produces consistently good results
    and does not depend strongly on the specific seed used during the optimization.

    For each independent training run, the validation loss and accuracy are recorded
    at every epoch. The epoch corresponding to the minimum validation loss is then
    selected, and its loss and accuracy are used to compute the mean and standard
    deviation across all seeds.

    Args:
        model_class (type): Model class to instantiate independently for each seed.
        train_loader (torch.utils.data.DataLoader): DataLoader used for training.
        val_loader (torch.utils.data.DataLoader): DataLoader used for validation.
        optimizer_class (type): Optimizer class used for training.
        optimizer_params (dict): Parameters passed to the optimizer constructor.
        loss_fn_class (type): Loss function class.
        scheduler_class (type): Learning-rate scheduler class.
        scheduler_params (dict): Parameters passed to the scheduler constructor.

    Returns:
        tuple:
            avg_loss (float): Mean of the minimum validation losses across seeds.
            avg_acc (float): Mean validation accuracy measured at the epoch of
                minimum validation loss for each seed.
            std_dev_loss (float): Standard deviation of the minimum validation
                losses across seeds.
            std_dev_acc (float): Standard deviation of the corresponding validation
                accuracies across seeds.
    """
    # let's create a dictionary to store the results of validation losses and accuracies
    results = {
        "losses": [],
        "accuracies": [],
    }
    
    # compute losses and accuracies
    for i, seed in enumerate(SEEDS):
        print(f"Seed test {i+1}/{len(SEEDS)} - seed = {seed}...")
        torch.manual_seed(seed)
        model = model_class().to(config.DEVICE)
        loss_fn = loss_fn_class()
        functions.fluctuation_driven_initialization(model, train_loader)                        
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = scheduler_class(optimizer, **scheduler_params)       

        loss_hist, acc_hist= [], []

        for _ in range(config.EPOCHS):
            loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)
            loss_test, acc_test = functions.evaluate(model, val_loader, loss_fn)
            scheduler.step()
            loss_hist.append(loss_test)
            acc_hist.append(acc_test)

        results["losses"].append(loss_hist)
        results["accuracies"].append(acc_hist)

    # plot the results and store them
    fig, ax = plt.subplots(1, 2, figsize = (12,4), squeeze=False)
    for i in range(len(results["losses"])):
        loss = results["losses"][i]
        acc = results["accuracies"][i]
        ax[0, 0].plot(loss)
        ax[0, 0].set_title("Validation losses")
        ax[0, 0].set_ylabel("Losses")
        ax[0, 1].plot(acc)
        ax[0, 1].set_title("Validation accuracies")
        ax[0, 1].set_ylabel("Accuracies")
    plt.tight_layout()

    # compute average loss and accuracies
    best_losses, best_accs, summary = [], [], []
    for losses, accs, seed in zip(results["losses"], results["accuracies"], SEEDS):
        best_epoch = losses.index(min(losses))
        best_losses.append(losses[best_epoch])
        best_accs.append(accs[best_epoch])
        summary.append({
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": losses[best_epoch],
                "val_accuracy_at_best_loss": accs[best_epoch],
                "final_val_loss": losses[-1],
                "final_val_accuracy": accs[-1]
            })
    avg_loss = sum(best_losses)/len(best_losses)
    avg_acc = sum(best_accs)/len(best_accs)
    std_dev_loss = torch.tensor(best_losses, dtype=torch.float).std().item()
    std_dev_acc = torch.tensor(best_accs, dtype=torch.float).std().item()

    print(f"average loss is: {avg_loss}±{std_dev_loss}")
    print(f"average accuracy is: {avg_acc}±{std_dev_acc}")

    save_robustness_results(
        test_name="different_seeds",
        model_class=model_class,
        results=results,
        summary=summary,
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        std_dev_loss=std_dev_loss,
        std_dev_acc=std_dev_acc,
        fig=fig,
        tested_parameter_name="seed",
        tested_parameter_values=SEEDS,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        loss_fn_class=loss_fn_class,
        scheduler_class=scheduler_class,
        scheduler_params=scheduler_params
    )

    plt.show()
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

def robustness_test_different_gaps(model_class, optimizer_class=config.OPTIMIZER, optimizer_params = config.OPTIMIZER_PARAMS, 
                                   loss_fn_class=config.LOSS_FN, scheduler_class = config.SCHEDULER, 
                                   scheduler_params = config.SCHEDULER_PARAMS, seed=config.SEED):
    """
    Tests the robustness of the training procedure across different temporal gaps.

    For each gap defined in `GAPS`, a new dataset is generated and a new model is
    trained from scratch using the same hyperparameter configuration and training seed.

    Changing the temporal gap allows us to evaluate whether the selected training
    configuration remains effective when the temporal distance between input A and
    input B is modified.

    For each independent training run, validation loss and accuracy are recorded at
    every epoch. The epoch corresponding to the minimum validation loss is selected
    and used to compare performance across the tested temporal gaps.

    Args:
        model_class (type): Model class to instantiate independently for each gap.
        optimizer_class (type): Optimizer class used for training.
        optimizer_params (dict): Parameters passed to the optimizer constructor.
        loss_fn_class (type): Loss function class.
        scheduler_class (type): Learning-rate scheduler class.
        scheduler_params (dict): Parameters passed to the scheduler constructor.
        seed (int): Seed used to control model initialization and training stochasticity.

    Returns:
        tuple:
            avg_loss (float): Mean of the minimum validation losses across tested gaps.
            avg_acc (float): Mean validation accuracy measured at the minimum-loss
                epoch for each tested gap.
            std_dev_loss (float): Standard deviation of the minimum validation losses
                across tested gaps.
            std_dev_acc (float): Standard deviation of the corresponding validation
                accuracies across tested gaps.
    """

    # let's create a dictionary to store the results of validation losses and accuracies
    results = {
        "losses": [],
        "accuracies": [],
    }
    
    # compute losses and accuracies
    for i, gap in enumerate(GAPS):
        # we neeed to create the dataset everytime with a different gap
        train_dataset, val_dataset = functions.generate_train_validation_dataset(gap = gap)
        train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)

        print(f"Gap test {i+1}/{len(GAPS)} - gap = {gap}...")
        torch.manual_seed(seed)
        model = model_class().to(config.DEVICE)
        loss_fn = loss_fn_class()
        functions.fluctuation_driven_initialization(model, train_loader)                        
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = scheduler_class(optimizer, **scheduler_params)       

        loss_hist, acc_hist= [], []

        for _ in range(config.EPOCHS):
            loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)
            loss_test, acc_test = functions.evaluate(model, val_loader, loss_fn)
            scheduler.step()
            loss_hist.append(loss_test)
            acc_hist.append(acc_test)

        results["losses"].append(loss_hist)
        results["accuracies"].append(acc_hist)

    # plot the results and store them
    fig, ax = plt.subplots(1, 2, figsize = (12,4), squeeze=False)
    for i in range(len(results["losses"])):
        loss = results["losses"][i]
        acc = results["accuracies"][i]
        gap = GAPS[i]
        ax[0, 0].plot(loss, label=f"GAP = {gap}")
        ax[0, 1].plot(acc, label=f"GAP = {gap}")
    ax[0, 0].set_title("Validation losses")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 1].set_title("Validation accuracies")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 0].legend()
    ax[0, 1].legend()
    fig.suptitle("Gap Robustness test")
    fig.tight_layout()

    # compute average loss and accuracies
    best_losses, best_accs, summary = [], [], []
    for losses, accs, gap in zip(results["losses"], results["accuracies"], GAPS):
        best_epoch = losses.index(min(losses))
        best_losses.append(losses[best_epoch])
        best_accs.append(accs[best_epoch])
        summary.append({
                "gap": gap,
                "best_epoch": best_epoch,
                "best_val_loss": losses[best_epoch],
                "val_accuracy_at_best_loss": accs[best_epoch],
                "final_val_loss": losses[-1],
                "final_val_accuracy": accs[-1]
            })
        
    # Plot the best validation loss and corresponding accuracy as a function of the temporal gap
    fig_gap, ax_gap = plt.subplots(1, 2, figsize=(12, 4))
    ax_gap[0].plot(GAPS, best_losses, marker="o")
    ax_gap[0].set_xlabel("Gap")
    ax_gap[0].set_ylabel("Best validation loss")
    ax_gap[0].set_title("Best validation loss vs GAP")
    ax_gap[1].plot(GAPS, best_accs, marker="o")
    ax_gap[1].set_xlabel("Gap")
    ax_gap[1].set_ylabel("Validation accuracy (%)")
    ax_gap[1].set_title("Validation accuracy vs GAP")
    fig_gap.suptitle("Performance across temporal gaps")
    fig_gap.tight_layout()

    avg_loss = sum(best_losses)/len(best_losses)
    avg_acc = sum(best_accs)/len(best_accs)
    std_dev_loss = torch.tensor(best_losses, dtype=torch.float).std().item()
    std_dev_acc = torch.tensor(best_accs, dtype=torch.float).std().item()

    print(f"average loss is: {avg_loss}±{std_dev_loss}")
    print(f"average accuracy is: {avg_acc}±{std_dev_acc}")

    save_robustness_results(
        test_name="different_gaps",
        model_class=model_class,
        results=results,
        summary=summary,
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        std_dev_loss=std_dev_loss,
        std_dev_acc=std_dev_acc,
        fig=fig,
        summary_fig=fig_gap,
        tested_parameter_name="gap",
        tested_parameter_values=GAPS,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        loss_fn_class=loss_fn_class,
        scheduler_class=scheduler_class,
        scheduler_params=scheduler_params
    )

    plt.show()
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

def save_robustness_results(
    test_name,
    model_class,
    results,
    summary,
    avg_loss,
    avg_acc,
    std_dev_loss,
    std_dev_acc,
    fig,
    tested_parameter_name,
    tested_parameter_values,
    summary_fig=None,
    optimizer_class=config.OPTIMIZER,
    optimizer_params=config.OPTIMIZER_PARAMS,
    loss_fn_class=config.LOSS_FN,
    scheduler_class=config.SCHEDULER,
    scheduler_params=config.SCHEDULER_PARAMS
):
    """
    Stores the results and configuration of a robustness test in a
    timestamped archive directory.

    Args:
        test_name (str): Name of the robustness test.
        model_class (type): Model class used in the experiment.
        results (dict): Complete epoch-by-epoch histories.
        summary (list): Summary metrics for each independent run.
        avg_loss (float): Mean best validation loss.
        avg_acc (float): Mean validation accuracy at the best-loss epoch.
        std_dev_loss (float): Standard deviation of best validation losses.
        std_dev_acc (float): Standard deviation of corresponding accuracies.
        fig (matplotlib.figure.Figure): Figure containing robustness curves.
        tested_parameter_name (str): Name of the parameter varied in the test.
        tested_parameter_values (list): Values tested for the selected parameter.
        summary_fig (matplotlib.figure.Figure | None): Optional summary figure.
        optimizer_class (type): Optimizer class.
        optimizer_params (dict): Optimizer parameters.
        loss_fn_class (type): Loss function class.
        scheduler_class (type): Scheduler class.
        scheduler_params (dict): Scheduler parameters.

    Returns:
        str: Path of the directory where results have been stored.
    """

    # create directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = os.path.join(
        "robustness_results",
        test_name,
        timestamp
    )
    os.makedirs(archive_dir, exist_ok=True)

    # metadata we want to save
    metadata = {
        "test_name": test_name,
        "date": timestamp,

        "model": model_class.__name__,
        "device": str(config.DEVICE),

        "epochs": config.EPOCHS,
        "hidden_layer": config.HIDDEN_LAYER,
        "beta": config.BETA,

        "optimizer": optimizer_class.__name__,
        "optimizer_params": optimizer_params,

        "scheduler": scheduler_class.__name__,
        "scheduler_params": scheduler_params,

        "loss_fn": loss_fn_class.__name__,

        "initialization": "fluctuation_driven",
        "fluct_xi": config.FLUCT_XI,

        "tested_parameter": tested_parameter_name,
        "tested_values": tested_parameter_values,

        "num_samples": config.NUM_SAMPLES,
        "train_split": config.TRAIN_SPLIT,

        "prob_0": config.PROB_0,
        "prob_1": config.PROB_1,

        "sign_a": config.SIGN_A,
        "sign_b": config.SIGN_B,
        "nominal_gap": config.GAP,
        "readout": config.READOUT,

        "results": {
            "avg_best_val_loss": avg_loss,
            "std_best_val_loss": std_dev_loss,
            "avg_val_accuracy": avg_acc,
            "std_val_accuracy": std_dev_acc
        }
    }

    # write metadata in a .json
    with open(
        os.path.join(archive_dir, "metadata.json"),
        "w"
    ) as file:
        json.dump(metadata, file, indent=4)

    # create csv with the results
    with open(
        os.path.join(archive_dir, "summary.csv"),
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary)

    # save complete epoch-by-epoch histories
    torch.save(
        {
            "tested_parameter": tested_parameter_name,
            "tested_values": tested_parameter_values,
            "losses": results["losses"],
            "accuracies": results["accuracies"]
        },
        os.path.join(archive_dir, "histories.pt")
    )

    # save training curves
    fig.savefig(
        os.path.join(archive_dir, "curves.png"),
        dpi=300,
        bbox_inches="tight"
    )

    # save optional summary figure
    if summary_fig is not None:
        summary_fig.savefig(
            os.path.join(archive_dir, "summary.png"),
            dpi=300,
            bbox_inches="tight"
        )

    print(
        f"\nRobustness results stored in:\n"
        f"{archive_dir}\n"
    )

    return archive_dir

## --- MAIN CODE ---

if __name__ == "__main__":

    # SEED robustness test
    # train_dataset, val_dataset = functions.generate_train_validation_dataset()
    # train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)
    # avg_loss, avg_acc, std_dev_loss, std_dev_acc = robustness_test_different_seeds(classes.XORSNN, train_loader, val_loader)

    # GAP robustness test
    avg_loss, avg_acc, std_dev_loss, std_dev_acc = robustness_test_different_gaps(classes.XORSNN)