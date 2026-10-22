"""
Robustness and Generalization Evaluation Suite for SNN Models.

Evaluates network stability and temporal memory under varying conditions:
- Seed Invariance: measures training reproducibility across independent seeds.
- Gap Scaling (Training): trains models from scratch over different temporal gaps (`GAPS_TRAIN`).
- Gap Scaling (Frozen Model): tests temporal memory persistence of a pre-trained model
  over varying gap lengths across multiple stochastic test datasets (`GAPS_TEST`, mean ± std).
- Gap Noise Analysis: measures sensitivity to disturbance spikes injected during the gap.
- Probability Sensitivity: evaluates resilience to variations in spike emission probabilities (P0, P1).
- Gap-Noise Interaction: evaluates 3D resilience surface crossing temporal GAP length and noise count.
- Cross-Gap Generalization: trains models across `GAPS_TRAIN` (with noise) and benchmarks each across `GAPS_TEST`.

Automatically logs metrics, curves, and summary CSVs under 'robustness/results/<study_name>/<model_name>/'.
Configurable via JSON inputs under 'robustness/inputs/'.
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

SEEDS = [i for i in range(1000, 1020)]
GAPS_TRAIN = [20, 30, 40, 50, 60, 70]
GAPS_TEST = [0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]
# Fixed disturbance spike positions inside GAP (Channel A: 0-29, Channel B: 30-59)
# Systematically sampled disturbance spikes spanning GAP [0, 30] (A: 0-29, B: 30-59)
NOISE = {
    # 1 Spike: no mixed possible (7 points per channel)
    1: [
        # Channel A
        (3,), (7,), (11,), (15,), (19,), (23,), (27,),
        # Channel B (+30)
        (33,), (37,), (41,), (45,), (49,), (53,), (57,),
    ],

    # 2 Spikes: 1 on A + 1 on B for mixed (7 points per line)
    2: [
        # Channel A
        (2, 4), (6, 8), (10, 12), (14, 16), (18, 20), (22, 24), (26, 28),
        # Channel B (+30)
        (32, 34), (36, 38), (40, 42), (44, 46), (48, 50), (52, 54), (56, 58),
        # Mixed (1 on A, 1 on B synchronous at time t)
        (3, 33), (7, 37), (11, 41), (15, 45), (19, 49), (23, 53), (27, 57),
    ],

    # 3 Spikes: 2 on A + 1 on B for mixed (6 points per line)
    3: [
        # Channel A
        (3, 5, 7), (7, 9, 11), (11, 13, 15), (15, 17, 19), (19, 21, 23), (23, 25, 27),
        # Channel B (+30)
        (33, 35, 37), (37, 39, 41), (41, 43, 45), (45, 47, 49), (49, 51, 53), (53, 55, 57),
        # Mixed (2 on A, 1 on B centered at time t)
        (4, 6, 35), (8, 10, 39), (12, 14, 43), (16, 18, 47), (20, 22, 51), (24, 26, 55),
    ],

    # 4 Spikes: 2 on A + 2 on B for mixed (6 points per line)
    4: [
        # Channel A
        (3, 5, 7, 9), (7, 9, 11, 13), (11, 13, 15, 17), (15, 17, 19, 21), (19, 21, 23, 25), (23, 25, 27, 29),
        # Channel B (+30)
        (33, 35, 37, 39), (37, 39, 41, 43), (41, 43, 45, 47), (45, 47, 49, 51), (49, 51, 53, 55), (53, 55, 57, 59),
        # Mixed (2 on A + 2 on B symmetric at time t)
        (4, 6, 34, 36), (8, 10, 38, 40), (12, 14, 42, 44), (16, 18, 46, 48), (20, 22, 50, 52), (24, 26, 54, 56),
    ],
}
PROB_0 = [round(x * 0.02, 2) for x in range(10, 21)]        # 0.20 to 0.40 (step 0.02)
PROB_1 = [round(x * 0.02, 2) for x in range(25, 36)]        # 0.50 to 0.70 (step 0.02)
GAP_NOISE_GAPS = [10, 15, 20, 25, 30, 35, 40]              # temporal gaps evaluated against noise
GAP_NOISE_COUNTS = [0, 1, 2, 3, 4]                          # noise spike counts injected in GAP

## --- Functions ---

# different training conditions

def robustness_train_seeds(model_class, train_loader, val_loader, optimizer_class=config.OPTIMIZER,
                                    optimizer_params = config.OPTIMIZER_PARAMS, loss_fn_class=config.LOSS_FN, 
                                    scheduler_class = config.SCHEDULER, scheduler_params = config.SCHEDULER_PARAMS,
                                    output_dir=None, show_plot=False):

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
        output_dir (str, optional): Target directory to save artifacts. If None, uses default timestamped path.
        show_plot (bool, optional): If True, opens interactive matplotlib window; closes figure otherwise.

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
        test_name="different_seeds_train",
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
        scheduler_params=scheduler_params,
        output_dir=output_dir
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

def robustness_train_gaps(model_class, optimizer_class=config.OPTIMIZER, optimizer_params=config.OPTIMIZER_PARAMS, 
                          loss_fn_class=config.LOSS_FN, scheduler_class=config.SCHEDULER, 
                          scheduler_params=config.SCHEDULER_PARAMS, train_seeds=SEEDS[:3],
                          noise=config.NOISE_AWARE, output_dir=None, show_plot=False):
    """
    Tests the robustness of the training procedure across different temporal gaps.

    For each gap in `GAPS_TRAIN`, independent models are trained from scratch across multiple
    random seeds (`train_seeds`) to separate gap difficulty from stochastic initialization
    and optimization variance.

    Args:
        model_class (type): Model class to instantiate independently for each gap.
        optimizer_class (type): Optimizer class used for training.
        optimizer_params (dict): Parameters passed to the optimizer constructor.
        loss_fn_class (type): Loss function class.
        scheduler_class (type): Learning-rate scheduler class.
        scheduler_params (dict): Parameters passed to the scheduler constructor.
        train_seeds (list): Seeds used for multi-run training per gap (default: first 3 seeds).
        noise (bool): If True, trains with multi-condition mixture noise; clean otherwise.
        output_dir (str, optional): Target directory to save artifacts. If None, uses default timestamped path.
        show_plot (bool, optional): If True, opens interactive matplotlib window; closes figures otherwise.

    Returns:
        tuple:
            avg_loss (float): Mean of minimum validation losses across tested gaps.
            avg_acc (float): Mean validation accuracy measured across tested gaps.
            std_dev_loss (float): Standard deviation of losses across gaps.
            std_dev_acc (float): Standard deviation of accuracies across gaps.
    """

    results = {
        "losses": [],
        "accuracies": [],
    }
    
    summary = []
    mean_best_losses, std_best_losses = [], []
    mean_val_accs, std_val_accs = [], []

    # compute losses and accuracies across gaps and training seeds
    for i, gap in enumerate(GAPS_TRAIN):
        gap_best_losses, gap_best_accs = [], []
        gap_final_losses, gap_final_accs = [], []

        for s_idx, s in enumerate(train_seeds):
            print(f"Gap train {i+1}/{len(GAPS_TRAIN)} (gap={gap}) - Seed {s_idx+1}/{len(train_seeds)} (seed={s})...")
            torch.manual_seed(s)
            train_dataset, val_dataset = functions.generate_train_validation_dataset(gap=gap, seed=s, noise=noise)
            train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)

            model = model_class().to(config.DEVICE)
            functions.fluctuation_driven_initialization(model, train_loader)
            loss_fn = loss_fn_class()
            optimizer = optimizer_class(model.parameters(), **optimizer_params)
            scheduler = scheduler_class(optimizer, **scheduler_params)

            loss_hist, acc_hist = [], []
            for _ in range(config.EPOCHS):
                loss_train, acc_train = functions.train(model, train_loader, optimizer, loss_fn)
                loss_val, acc_val = functions.evaluate(model, val_loader, loss_fn)
                scheduler.step()
                loss_hist.append(loss_val)
                acc_hist.append(acc_val)

            results["losses"].append(loss_hist)
            results["accuracies"].append(acc_hist)

            best_ep = loss_hist.index(min(loss_hist))
            gap_best_losses.append(loss_hist[best_ep])
            gap_best_accs.append(acc_hist[best_ep])
            gap_final_losses.append(loss_hist[-1])
            gap_final_accs.append(acc_hist[-1])

        m_loss = sum(gap_best_losses) / len(gap_best_losses)
        s_loss = torch.tensor(gap_best_losses).std().item() if len(gap_best_losses) > 1 else 0.0
        m_acc = sum(gap_best_accs) / len(gap_best_accs)
        s_acc = torch.tensor(gap_best_accs).std().item() if len(gap_best_accs) > 1 else 0.0

        mean_best_losses.append(m_loss)
        std_best_losses.append(s_loss)
        mean_val_accs.append(m_acc)
        std_val_accs.append(s_acc)

        summary.append({
            "gap": gap,
            "mean_best_val_loss": m_loss,
            "std_best_val_loss": s_loss,
            "mean_val_accuracy": m_acc,
            "std_val_accuracy": s_acc,
            "min_val_accuracy": min(gap_best_accs),
            "max_val_accuracy": max(gap_best_accs),
            "num_seeds": len(train_seeds),
            "noise": noise
        })

    # Relative degradation Delta A against nominal GAP
    nom_idx = GAPS_TRAIN.index(config.GAP) if config.GAP in GAPS_TRAIN else None
    nom_acc = mean_val_accs[nom_idx] if nom_idx is not None else 100.0
    for row in summary:
        row["delta_accuracy"] = row["mean_val_accuracy"] - nom_acc

    # Plot validation loss and accuracy histories across all runs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    run_idx = 0
    for gap in GAPS_TRAIN:
        for s in train_seeds:
            loss = results["losses"][run_idx]
            acc = results["accuracies"][run_idx]
            run_idx += 1
            ax[0, 0].plot(loss, alpha=0.5, label=f"GAP={gap}" if s == train_seeds[0] else "")
            ax[0, 1].plot(acc, alpha=0.5, label=f"GAP={gap}" if s == train_seeds[0] else "")
    ax[0, 0].set_title("Validation losses across training runs")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 1].set_title("Validation accuracies across training runs")
    ax[0, 1].set_ylabel("Accuracy (%)")
    ax[0, 0].legend()
    ax[0, 1].legend()
    fig.suptitle("Gap Robustness Training Curves")
    fig.tight_layout()

    # Plot mean and std error bars across independent training seeds
    fig_gap, ax_gap = plt.subplots(1, 2, figsize=(12, 4))
    ax_gap[0].errorbar(GAPS_TRAIN, mean_best_losses, yerr=std_best_losses, fmt="o-", capsize=3)
    ax_gap[0].set_xlabel("Gap (timesteps)")
    ax_gap[0].set_ylabel("Best validation loss")
    ax_gap[0].set_title("Validation loss vs GAP (mean ± std across seeds)")
    ax_gap[1].errorbar(GAPS_TRAIN, mean_val_accs, yerr=std_val_accs, fmt="o-", capsize=3)
    ax_gap[1].set_xlabel("Gap (timesteps)")
    ax_gap[1].set_ylabel("Validation accuracy (%)")
    ax_gap[1].set_title("Validation accuracy vs GAP (mean ± std across seeds)")
    fig_gap.suptitle("Training performance across temporal gaps")
    fig_gap.tight_layout()

    avg_loss = sum(mean_best_losses) / len(mean_best_losses)
    avg_acc = sum(mean_val_accs) / len(mean_val_accs)
    std_dev_loss = torch.tensor(mean_best_losses, dtype=torch.float).std().item()
    std_dev_acc = torch.tensor(mean_val_accs, dtype=torch.float).std().item()

    print(f"Average loss across gaps: {avg_loss:.4f}±{std_dev_loss:.4f}")
    print(f"Average accuracy across gaps: {avg_acc:.2f}%±{std_dev_acc:.2f}%")

    save_robustness_results(
        test_name="different_gaps_train",
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
        tested_parameter_values=GAPS_TRAIN,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        loss_fn_class=loss_fn_class,
        scheduler_class=scheduler_class,
        scheduler_params=scheduler_params,
        output_dir=output_dir
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        if fig_gap is not None:
            plt.close(fig_gap)
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

# different test conditions

def robustness_test_gaps(model, loss_fn_class=config.LOSS_FN, seed=config.SEED, noise=config.NOISE_AWARE, output_dir=None, show_plot=False):
    """
    Tests the robustness of the test procedure across different temporal gaps.

    For each gap defined in `GAPS_TEST`, a new dataset is generated and, selecting a model
    inside the "models" directory we can test the robustness if we vary the gap length.

    Changing the temporal gap allows us to evaluate whether the selected training
    configuration remains effective when the temporal distance between input A and
    input B is modified.

    For each independent test run, validation loss and accuracy are recorded at
    every epoch. The epoch corresponding to the minimum validation loss is selected
    and used to compare performance across the tested temporal gaps.

    Args:
        model (torch.nn.Module): Pre-trained model to evaluate.
        loss_fn_class (type): Loss function class.
        seed (int): Seed used to control model initialization and training stochasticity.
        noise (bool): If True, tests with multi-condition mixture noise; clean otherwise.
        output_dir (str, optional): Target directory to save artifacts. If None, uses default timestamped path.
        show_plot (bool, optional): If True, opens interactive matplotlib window; closes figures otherwise.

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

    model.to(config.DEVICE)

    # let's create a dictionary to store the results of validation losses and accuracies
    results = {
        "losses": [],
        "accuracies": [],
    }
    
    # compute losses and accuracies
    for i, gap in enumerate(GAPS_TEST):
        loss_hist, acc_hist= [], []
        for N in range(config.N_ROBUSTNESS_DATASET):
            # we neeed to create the dataset everytime with a different gap
            test_dataset = functions.generate_test_dataset(gap=gap, seed=seed + N, noise=noise)
            test_loader = functions.generate_test_dataloader(test_dataset)

            print(f"Gap test {i * config.N_ROBUSTNESS_DATASET + N + 1}/{len(GAPS_TEST)*config.N_ROBUSTNESS_DATASET} - gap = {gap}...")
            loss_fn = loss_fn_class()     

            # single pass: frozen model on static test set is deterministic (epochs are only for training)
            loss_test, acc_test = functions.evaluate(model, test_loader, loss_fn)
            loss_hist.append(loss_test)
            acc_hist.append(acc_test)

        results["losses"].append(loss_hist)
        results["accuracies"].append(acc_hist)

    # plot the results and store them
    fig, ax = plt.subplots(1, 2, figsize = (12,4), squeeze=False)
    for i in range(len(results["losses"])):
        loss = results["losses"][i]
        acc = results["accuracies"][i]
        gap = GAPS_TEST[i]
        ax[0, 0].plot(loss, label=f"GAP = {gap}")
        ax[0, 1].plot(acc, label=f"GAP = {gap}")
    ax[0, 0].set_title("Test losses")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 1].set_title("Test accuracies")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 0].legend()
    ax[0, 1].legend()
    fig.suptitle("Gap Robustness test")
    fig.tight_layout()

    mean_losses, std_losses = [], []
    mean_accs, std_accs = [], []
    summary = []
    for losses, accs, gap in zip(results["losses"], results["accuracies"], GAPS_TEST):
        m_loss = sum(losses) / len(losses)
        s_loss = torch.tensor(losses).std().item()
        m_acc = sum(accs) / len(accs)
        s_acc = torch.tensor(accs).std().item()

        mean_losses.append(m_loss)
        std_losses.append(s_loss)
        mean_accs.append(m_acc)
        std_accs.append(s_acc)

        summary.append({
            "gap": gap,
            "mean_test_loss": m_loss,
            "std_test_loss": s_loss,
            "mean_test_accuracy": m_acc,
            "std_test_accuracy": s_acc,
            "min_test_accuracy": min(accs),                                    # worst-case dataset accuracy for this gap
            "max_test_accuracy": max(accs),                                    # best-case dataset accuracy for this gap
            "noise": noise
        })

    # Relative degradation Delta A against nominal GAP
    nom_idx = GAPS_TEST.index(config.GAP) if config.GAP in GAPS_TEST else None
    nom_acc = mean_accs[nom_idx] if nom_idx is not None else 100.0
    for row in summary:
        row["delta_accuracy"] = row["mean_test_accuracy"] - nom_acc
        
    # Plot the best validation loss and corresponding accuracy as a function of the temporal gap
    fig_gap, ax_gap = plt.subplots(1, 2, figsize=(12, 4))
    ax_gap[0].plot(GAPS_TEST, mean_losses, marker="o")
    ax_gap[0].set_xlabel("Gap")
    ax_gap[0].set_ylabel("Mean test loss")
    ax_gap[0].set_title("Mean test loss vs GAP")
    ax_gap[1].plot(GAPS_TEST, mean_accs, marker="o")
    ax_gap[1].set_xlabel("Gap")
    ax_gap[1].set_ylabel("Test accuracy (%)")
    ax_gap[1].set_title("Test accuracy vs GAP")
    fig_gap.suptitle("Performance across temporal gaps")
    fig_gap.tight_layout()

    avg_loss = sum(mean_losses)/len(mean_losses)
    avg_acc = sum(mean_accs)/len(mean_accs)
    std_dev_loss = torch.tensor(mean_losses, dtype=torch.float).std().item()
    std_dev_acc = torch.tensor(mean_accs, dtype=torch.float).std().item()

    print(f"average loss is: {avg_loss}±{std_dev_loss}")
    print(f"average accuracy is: {avg_acc}±{std_dev_acc}")

    save_robustness_results(
        test_name="different_gaps_test",
        model_class=model.__class__,
        results=results,
        summary=summary,
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        std_dev_loss=std_dev_loss,
        std_dev_acc=std_dev_acc,
        fig=fig,
        summary_fig=fig_gap,
        tested_parameter_name="gap",
        tested_parameter_values=GAPS_TEST,
        loss_fn_class=loss_fn_class,
        output_dir=output_dir
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        if fig_gap is not None:
            plt.close(fig_gap)
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

def robustness_test_noise(model, loss_fn_class=config.LOSS_FN, gap=config.GAP, seed=config.SEED, output_dir=None, show_plot=False):
    """
    Evaluates model robustness against disturbance spikes injected into the temporal GAP.

    For each configuration in `NOISE` (1 to 4 disturbance spikes), multiple test datasets 
    are generated across independent seeds. Evaluates performance as a function of the 
    temporal position (early, mid, late) and channel assignment (Channel A vs Channel B).

    Args:
        model (torch.nn.Module): Pre-trained frozen model to evaluate.
        loss_fn_class (type): Criterion class used for loss computation.
        seed (int): Base seed for deterministic test set generation.
        gap (int): Temporal duration of the GAP between input A and input B.
        output_dir (str, optional): Target directory to save artifacts. If None, uses default timestamped path.
        show_plot (bool, optional): If True, opens interactive matplotlib window; closes figures otherwise.

    Returns:
        tuple:
            avg_loss (float): Global mean test loss across all noise configurations.
            avg_acc (float): Global mean test accuracy across all noise configurations.
            std_dev_loss (float): Standard deviation of test loss across configurations.
            std_dev_acc (float): Standard deviation of test accuracy across configurations.
    """
    if gap != 30:
        raise ValueError(
            f"The positional NOISE dictionary is geometrically calibrated for GAP=30 (got {gap}). "
            f"For variable gap tests with noise, use robustness_test_gap_noise()."
        )

    model.to(config.DEVICE)

    # dictionary storing full seed histories, temporal metrics and channel labels
    results = {
        "losses": {1: [], 2: [], 3: [], 4: []},
        "accuracies": {1: [], 2: [], 3: [], 4: []},
        "metrics": {1: [], 2: [], 3: [], 4: []},
        "channels": {1: [], 2: [], 3: [], 4: []},
    }

    total_runs = sum(len(v) for v in NOISE.values()) * config.N_ROBUSTNESS_DATASET
    run_counter = 0

    # inference across noise configurations and independent seeds
    for i in range(1, 5):
        for j, spikes in enumerate(NOISE[i]):
            loss_hist, acc_hist = [], []
            for N in range(config.N_ROBUSTNESS_DATASET):
                run_counter += 1
                test_dataset = functions.generate_test_dataset(gap=gap, seed=seed + N, noise_spikes=spikes)
                test_loader = functions.generate_test_dataloader(test_dataset)

                print(f"Noise test {run_counter}/{total_runs} - {i} spikes: {spikes}...")
                loss_fn = loss_fn_class()

                # frozen model single pass evaluation
                loss_test, acc_test = functions.evaluate(model, test_loader, loss_fn)
                loss_hist.append(loss_test)
                acc_hist.append(acc_test)

            # determine channel assignment and mean timestep inside GAP
            channel = "A" if all(s < gap for s in spikes) else ("B" if all(s >= gap for s in spikes) else "Mixed")
            t_metric = sum(s if s < gap else (s - gap) for s in spikes) / len(spikes)

            results["losses"][i].append(loss_hist)           # full history across N seeds
            results["accuracies"][i].append(acc_hist)        # full history across N seeds
            results["metrics"][i].append(t_metric)           # mean timestep in GAP [0, 30]
            results["channels"][i].append(channel)          # channel label ('A', 'B', 'Mixed')

    # unified analysis, summary compilation and detailed curves plotting
    all_mean_losses, all_mean_accs = [], []
    summary = []
    fig, ax = plt.subplots(4, 2, figsize=(12, 16), squeeze=False)

    for i in range(1, 5):
        # compute mean values and variances of losses and accuracies
        mean_l = [sum(h) / len(h) for h in results["losses"][i]]
        std_l  = [torch.tensor(h).std().item() for h in results["losses"][i]]
        mean_a = [sum(h) / len(h) for h in results["accuracies"][i]]
        std_a  = [torch.tensor(h).std().item() for h in results["accuracies"][i]]

        # similar to append but we concatenate every number in a unique list
        all_mean_losses.extend(mean_l)
        all_mean_accs.extend(mean_a)

        # populate summary records reusing computed values
        for j, spikes in enumerate(NOISE[i]):
            summary.append({
                "num_spikes": i,
                "spike_positions": str(spikes),
                "channel": results["channels"][i][j],
                "time_in_gap": results["metrics"][i][j],
                "mean_test_loss": mean_l[j],
                "std_test_loss": std_l[j],
                "mean_test_accuracy": mean_a[j],
                "std_test_accuracy": std_a[j],
                "delta_accuracy": mean_a[j] - 100.0,            # relative degradation from clean nominal baseline (100%)
                "min_test_accuracy": min(results["accuracies"][i][j]),
                "max_test_accuracy": max(results["accuracies"][i][j]),
            })

        # plot channel-separated curves with standard deviation bars
        for col, (vals, stds, ylabel) in enumerate([(mean_l, std_l, "Loss"), (mean_a, std_a, "Accuracy (%)")]):
            curr_ax = ax[i - 1, col]
            for ch_label, marker, style in [("A", "o-", "Channel A"), ("B", "s--", "Channel B"), ("Mixed", "^", "Mixed A+B")]:
                idxs = [k for k, ch in enumerate(results["channels"][i]) if ch == ch_label]
                if idxs:
                    curr_ax.errorbar([results["metrics"][i][k] for k in idxs], [vals[k] for k in idxs],
                                     yerr=[stds[k] for k in idxs], fmt=marker, label=style, capsize=3)
            curr_ax.set_title(f"{ylabel} - {i} Spikes")
            curr_ax.set_xlabel("Time in GAP (timesteps)")
            curr_ax.set_ylabel(ylabel)
            curr_ax.legend()

    fig.suptitle("Spike Robustness Test across GAP")
    fig.tight_layout()

    # summary figure: performance scaling as a function of the disturbance spike count
    fig_noise, ax_noise = plt.subplots(1, 2, figsize=(12, 4))
    spike_counts = list(NOISE.keys())
    spike_mean_losses = [sum(m["mean_test_loss"] for m in summary if m["num_spikes"] == k) / len([m for m in summary if m["num_spikes"] == k]) for k in spike_counts]
    spike_mean_accs   = [sum(m["mean_test_accuracy"] for m in summary if m["num_spikes"] == k) / len([m for m in summary if m["num_spikes"] == k]) for k in spike_counts]

    ax_noise[0].plot(spike_counts, spike_mean_losses, marker="o")
    ax_noise[0].set_xlabel("Noise Spikes")
    ax_noise[0].set_ylabel("Mean test loss")
    ax_noise[0].set_title("Mean test loss vs Noise Spikes")
    ax_noise[0].set_xticks(spike_counts)
    ax_noise[1].plot(spike_counts, spike_mean_accs, marker="o")
    ax_noise[1].set_xlabel("Noise Spikes")
    ax_noise[1].set_ylabel("Test accuracy (%)")
    ax_noise[1].set_title("Test accuracy vs Noise Spikes")
    ax_noise[1].set_xticks(spike_counts)
    fig_noise.suptitle("Performance across noise spikes count")
    fig_noise.tight_layout()

    # compute global statistics across all tested conditions
    avg_loss = sum(all_mean_losses) / len(all_mean_losses)
    avg_acc  = sum(all_mean_accs) / len(all_mean_accs)
    std_dev_loss = torch.tensor(all_mean_losses, dtype=torch.float).std().item()
    std_dev_acc  = torch.tensor(all_mean_accs, dtype=torch.float).std().item()

    print(f"average loss is: {avg_loss}±{std_dev_loss}")
    print(f"average accuracy is: {avg_acc}±{std_dev_acc}")

    # archive results, metadata, summary CSV, and both figures
    save_robustness_results(
        test_name="different_noise_test",
        model_class=model.__class__,
        results=results,
        summary=summary,
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        std_dev_loss=std_dev_loss,
        std_dev_acc=std_dev_acc,
        fig=fig,
        summary_fig=fig_noise,
        tested_parameter_name="noise_spikes",
        tested_parameter_values=spike_counts,
        loss_fn_class=loss_fn_class,
        output_dir=output_dir
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        if fig_noise is not None:
            plt.close(fig_noise)
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

def robustness_test_probabilities(model, loss_fn_class=config.LOSS_FN, gap=config.GAP, seed=config.SEED, output_dir=None, show_plot=False):
    """
    Tests the theoretical resilience of a pre-trained frozen model against variations
    in spike generation probabilities (prob_0 for bit 0, prob_1 for bit 1).

    Evaluates performance over a 2D grid (PROB_0 x PROB_1) across multiple stochastic
    test datasets, generating contour surfaces for test loss and accuracy.

    Args:
        model (torch.nn.Module): Pre-trained model to evaluate.
        loss_fn_class (type): Loss function class.
        gap (int): Temporal gap between input tokens.
        seed (int): Base seed for dataset generation stochasticity.
        output_dir (str, optional): Target directory to save artifacts. If None, uses default timestamped path.
        show_plot (bool, optional): If True, opens interactive matplotlib window; closes figures otherwise.

    Returns:
        tuple:
            avg_loss (float): Mean test loss across all tested probability pairs.
            avg_acc (float): Mean test accuracy across all tested probability pairs.
            std_dev_loss (float): Standard deviation of test loss across pairs.
            std_dev_acc (float): Standard deviation of test accuracy across pairs.
    """

    model.to(config.DEVICE)

    results = {
        "losses": [],
        "accuracies": [],
    }

    # 2D matrices for contour plotting: shape (len(PROB_1), len(PROB_0)) -> Y: prob_1, X: prob_0
    z_loss = torch.zeros((len(PROB_1), len(PROB_0)))                    # grid for mean loss surface
    z_acc = torch.zeros((len(PROB_1), len(PROB_0)))                     # grid for mean accuracy surface
    summary = []

    tot_test = len(PROB_0) * len(PROB_1) * config.N_ROBUSTNESS_DATASET

    # compute losses and accuracies across the 2D grid
    for i, prob_0 in enumerate(PROB_0):
        for j, prob_1 in enumerate(PROB_1):
            loss_hist, acc_hist = [], []
            for N in range(config.N_ROBUSTNESS_DATASET):
                test_dataset = functions.generate_test_dataset(
                    gap=gap, prob_0=prob_0, prob_1=prob_1, seed=seed + N
                )
                test_loader = functions.generate_test_dataloader(test_dataset)

                test_idx = i * len(PROB_1) * config.N_ROBUSTNESS_DATASET + j * config.N_ROBUSTNESS_DATASET + N + 1
                print(f"Probability test {test_idx}/{tot_test} - Probs -> 0: {prob_0} - 1: {prob_1}")

                loss_fn = loss_fn_class()
                loss_test, acc_test = functions.evaluate(model, test_loader, loss_fn)
                loss_hist.append(loss_test)
                acc_hist.append(acc_test)

            results["losses"].append(loss_hist)
            results["accuracies"].append(acc_hist)

            m_loss = sum(loss_hist) / len(loss_hist)
            s_loss = torch.tensor(loss_hist).std().item() if len(loss_hist) > 1 else 0.0
            m_acc = sum(acc_hist) / len(acc_hist)
            s_acc = torch.tensor(acc_hist).std().item() if len(acc_hist) > 1 else 0.0

            z_loss[j, i] = m_loss                                       # row j: prob_1 (Y), col i: prob_0 (X)
            z_acc[j, i] = m_acc

            summary.append({
                "prob_0": prob_0,
                "prob_1": prob_1,
                "mean_test_loss": m_loss,
                "std_test_loss": s_loss,
                "mean_test_accuracy": m_acc,
                "std_test_accuracy": s_acc,
                "min_test_accuracy": min(acc_hist),
                "max_test_accuracy": max(acc_hist)
            })

    # 3D grid mesh for surface and contour plotting
    p0_grid, p1_grid = torch.meshgrid(torch.tensor(PROB_0), torch.tensor(PROB_1), indexing="xy")

    # Nominal point coordinates
    nom_i, nom_j = PROB_0.index(config.PROB_0), PROB_1.index(config.PROB_1)
    nom_loss, nom_acc = z_loss[nom_j, nom_i].item(), z_acc[nom_j, nom_i].item()
    for row in summary:
        row["delta_accuracy"] = row["mean_test_accuracy"] - nom_acc

    # 3D figures with contour lines plotted directly on the 3D elevation
    fig = plt.figure(figsize=(15, 6))

    # Loss 3D surface + 3D contour lines
    ax_loss = fig.add_subplot(1, 2, 1, projection="3d")
    surf_loss = ax_loss.plot_surface(
        p0_grid.numpy(), p1_grid.numpy(), z_loss.numpy(),
        cmap="viridis", alpha=0.65, edgecolor="none"                    # translucent 3D terrain
    )
    c_loss = ax_loss.contour(
        p0_grid.numpy(), p1_grid.numpy(), z_loss.numpy(),
        levels=12, colors="black", linewidths=1.2                       # contour curves at actual Z height
    )
    ax_loss.scatter([config.PROB_0], [config.PROB_1], [nom_loss], color="red", s=130, label="Nominal (0.3, 0.6)")
    ax_loss.set_title("Test Loss 3D Contour Surface")
    ax_loss.set_xlabel("P0 (Bit 0 prob)")
    ax_loss.set_ylabel("P1 (Bit 1 prob)")
    ax_loss.set_zlabel("Mean Test Loss")
    fig.colorbar(surf_loss, ax=ax_loss, shrink=0.5, aspect=10, pad=0.1, label="Loss")
    ax_loss.legend()

    # Accuracy 3D surface + 3D contour lines
    ax_acc = fig.add_subplot(1, 2, 2, projection="3d")
    surf_acc = ax_acc.plot_surface(
        p0_grid.numpy(), p1_grid.numpy(), z_acc.numpy(),
        cmap="magma", alpha=0.65, edgecolor="none"                      # translucent 3D terrain
    )
    c_acc = ax_acc.contour(
        p0_grid.numpy(), p1_grid.numpy(), z_acc.numpy(),
        levels=12, colors="black", linewidths=1.2                       # contour curves at actual Z height
    )
    ax_acc.scatter([config.PROB_0], [config.PROB_1], [nom_acc], color="cyan", s=130, label="Nominal (0.3, 0.6)")
    ax_acc.set_title("Test Accuracy 3D Contour Surface")
    ax_acc.set_xlabel("P0 (Bit 0 prob)")
    ax_acc.set_ylabel("P1 (Bit 1 prob)")
    ax_acc.set_zlabel("Test Accuracy (%)")
    fig.colorbar(surf_acc, ax=ax_acc, shrink=0.5, aspect=10, pad=0.1, label="Accuracy (%)")
    ax_acc.legend()

    fig.suptitle("Spike Generation Probability Robustness (3D Contours)")
    fig.tight_layout()

    all_mean_losses = [row["mean_test_loss"] for row in summary]
    all_mean_accs = [row["mean_test_accuracy"] for row in summary]
    avg_loss = sum(all_mean_losses) / len(all_mean_losses)
    avg_acc = sum(all_mean_accs) / len(all_mean_accs)
    std_dev_loss = torch.tensor(all_mean_losses, dtype=torch.float).std().item()
    std_dev_acc = torch.tensor(all_mean_accs, dtype=torch.float).std().item()

    print(f"Average loss across (P0, P1) grid: {avg_loss:.4f}±{std_dev_loss:.4f}")
    print(f"Average accuracy across (P0, P1) grid: {avg_acc:.2f}%±{std_dev_acc:.2f}%")

    save_robustness_results(
        test_name="different_probabilities_test",
        model_class=model.__class__,
        results=results,
        summary=summary,
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        std_dev_loss=std_dev_loss,
        std_dev_acc=std_dev_acc,
        fig=fig,
        tested_parameter_name="probabilities",
        tested_parameter_values={"prob_0": PROB_0, "prob_1": PROB_1},
        loss_fn_class=loss_fn_class,
        output_dir=output_dir
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

def robustness_test_gap_noise(model, loss_fn_class=config.LOSS_FN, seed=config.SEED, output_dir=None, show_plot=False):
    """
    Tests the theoretical resilience of a pre-trained frozen model across simultaneous
    variations in temporal gap duration and disturbance noise spike count.

    Evaluates performance over a 2D parameter grid (GAP_NOISE_GAPS x GAP_NOISE_COUNTS)
    where noise spikes are stochastically distributed in the gap across multiple
    independent test datasets. Renders a 3D surface plot with elevation contour lines.

    Args:
        model (torch.nn.Module): Pre-trained model to evaluate.
        loss_fn_class (type): Loss function class.
        seed (int): Base seed for dataset generation stochasticity.
        output_dir (str, optional): Target directory to save artifacts. If None, uses default timestamped path.
        show_plot (bool, optional): If True, opens interactive matplotlib window; closes figures otherwise.

    Returns:
        tuple:
            avg_loss (float): Mean test loss across all tested (gap, noise) conditions.
            avg_acc (float): Mean test accuracy across all tested (gap, noise) conditions.
            std_dev_loss (float): Standard deviation of test loss across conditions.
            std_dev_acc (float): Standard deviation of test accuracy across conditions.
    """

    model.to(config.DEVICE)

    results = {
        "losses": [],
        "accuracies": [],
    }

    # 2D matrices for 3D surface: shape (len(GAP_NOISE_COUNTS), len(GAP_NOISE_GAPS)) -> Y: noise, X: gap
    z_loss = torch.zeros((len(GAP_NOISE_COUNTS), len(GAP_NOISE_GAPS)))
    z_acc = torch.zeros((len(GAP_NOISE_COUNTS), len(GAP_NOISE_GAPS)))
    summary = []

    tot_test = len(GAP_NOISE_GAPS) * len(GAP_NOISE_COUNTS) * config.N_ROBUSTNESS_DATASET

    # compute losses and accuracies across the (gap, noise) grid
    for i, gap in enumerate(GAP_NOISE_GAPS):
        for j, noise in enumerate(GAP_NOISE_COUNTS):
            loss_hist, acc_hist = [], []
            for N in range(config.N_ROBUSTNESS_DATASET):
                test_dataset = functions.generate_test_dataset(
                    gap=gap, noise_spikes=noise, seed=seed + N
                )
                test_loader = functions.generate_test_dataloader(test_dataset)

                test_idx = i * len(GAP_NOISE_COUNTS) * config.N_ROBUSTNESS_DATASET + j * config.N_ROBUSTNESS_DATASET + N + 1
                print(f"Gap-Noise test {test_idx}/{tot_test} - Gap: {gap} - Noise spikes: {noise}...")

                loss_fn = loss_fn_class()
                loss_test, acc_test = functions.evaluate(model, test_loader, loss_fn)
                loss_hist.append(loss_test)
                acc_hist.append(acc_test)

            results["losses"].append(loss_hist)
            results["accuracies"].append(acc_hist)

            m_loss = sum(loss_hist) / len(loss_hist)
            s_loss = torch.tensor(loss_hist).std().item() if len(loss_hist) > 1 else 0.0
            m_acc = sum(acc_hist) / len(acc_hist)
            s_acc = torch.tensor(acc_hist).std().item() if len(acc_hist) > 1 else 0.0

            z_loss[j, i] = m_loss                                       # row j: noise (Y), col i: gap (X)
            z_acc[j, i] = m_acc

            summary.append({
                "gap": gap,
                "noise_spikes": noise,
                "mean_test_loss": m_loss,
                "std_test_loss": s_loss,
                "mean_test_accuracy": m_acc,
                "std_test_accuracy": s_acc,
                "min_test_accuracy": min(acc_hist),
                "max_test_accuracy": max(acc_hist)
            })

    # 3D grid mesh for surface and contour plotting
    gap_grid, noise_grid = torch.meshgrid(
        torch.tensor(GAP_NOISE_GAPS, dtype=torch.float),
        torch.tensor(GAP_NOISE_COUNTS, dtype=torch.float),
        indexing="xy"
    )

    # Nominal point coordinates if within tested grid
    has_nominal = (config.GAP in GAP_NOISE_GAPS) and (config.NOISE_SPIKES in GAP_NOISE_COUNTS)
    if has_nominal:
        nom_i = GAP_NOISE_GAPS.index(config.GAP)
        nom_j = GAP_NOISE_COUNTS.index(config.NOISE_SPIKES)
        nom_loss = z_loss[nom_j, nom_i].item()
        nom_acc = z_acc[nom_j, nom_i].item()

    nom_acc_val = nom_acc if has_nominal else 100.0
    for row in summary:
        row["delta_accuracy"] = row["mean_test_accuracy"] - nom_acc_val

    # 3D figures with contour lines plotted directly on the 3D elevation
    fig = plt.figure(figsize=(15, 6))

    # Loss 3D surface + 3D contour lines
    ax_loss = fig.add_subplot(1, 2, 1, projection="3d")
    surf_loss = ax_loss.plot_surface(
        gap_grid.numpy(), noise_grid.numpy(), z_loss.numpy(),
        cmap="viridis", alpha=0.65, edgecolor="none"                    # translucent 3D terrain
    )
    ax_loss.contour(
        gap_grid.numpy(), noise_grid.numpy(), z_loss.numpy(),
        levels=12, colors="black", linewidths=1.2                       # contour curves at actual Z height
    )
    if has_nominal:
        ax_loss.scatter([config.GAP], [config.NOISE_SPIKES], [nom_loss], color="red", s=130, label=f"Nominal ({config.GAP}, {config.NOISE_SPIKES})")
    ax_loss.set_title("Test Loss 3D Surface (GAP vs Noise)")
    ax_loss.set_xlabel("GAP Duration (timesteps)")
    ax_loss.set_ylabel("Noise Spikes in GAP")
    ax_loss.set_zlabel("Mean Test Loss")
    ax_loss.set_xticks(GAP_NOISE_GAPS)
    ax_loss.set_yticks(GAP_NOISE_COUNTS)
    fig.colorbar(surf_loss, ax=ax_loss, shrink=0.5, aspect=10, pad=0.1, label="Loss")
    if has_nominal:
        ax_loss.legend()

    # Accuracy 3D surface + 3D contour lines
    ax_acc = fig.add_subplot(1, 2, 2, projection="3d")
    surf_acc = ax_acc.plot_surface(
        gap_grid.numpy(), noise_grid.numpy(), z_acc.numpy(),
        cmap="magma", alpha=0.65, edgecolor="none"                      # translucent 3D terrain
    )
    ax_acc.contour(
        gap_grid.numpy(), noise_grid.numpy(), z_acc.numpy(),
        levels=12, colors="black", linewidths=1.2                       # contour curves at actual Z height
    )
    if has_nominal:
        ax_acc.scatter([config.GAP], [config.NOISE_SPIKES], [nom_acc], color="cyan", s=130, label=f"Nominal ({config.GAP}, {config.NOISE_SPIKES})")
    ax_acc.set_title("Test Accuracy 3D Surface (GAP vs Noise)")
    ax_acc.set_xlabel("GAP Duration (timesteps)")
    ax_acc.set_ylabel("Noise Spikes in GAP")
    ax_acc.set_zlabel("Test Accuracy (%)")
    ax_acc.set_xticks(GAP_NOISE_GAPS)
    ax_acc.set_yticks(GAP_NOISE_COUNTS)
    fig.colorbar(surf_acc, ax=ax_acc, shrink=0.5, aspect=10, pad=0.1, label="Accuracy (%)")
    if has_nominal:
        ax_acc.legend()

    fig.suptitle("Temporal GAP vs Noise Robustness (3D Contours)")
    fig.tight_layout()

    all_mean_losses = [row["mean_test_loss"] for row in summary]
    all_mean_accs = [row["mean_test_accuracy"] for row in summary]
    avg_loss = sum(all_mean_losses) / len(all_mean_losses)
    avg_acc = sum(all_mean_accs) / len(all_mean_accs)
    std_dev_loss = torch.tensor(all_mean_losses, dtype=torch.float).std().item()
    std_dev_acc = torch.tensor(all_mean_accs, dtype=torch.float).std().item()

    print(f"Average loss across (GAP, Noise) grid: {avg_loss:.4f}±{std_dev_loss:.4f}")
    print(f"Average accuracy across (GAP, Noise) grid: {avg_acc:.2f}%±{std_dev_acc:.2f}%")

    save_robustness_results(
        test_name="different_gap_noise_test",
        model_class=model.__class__,
        results=results,
        summary=summary,
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        std_dev_loss=std_dev_loss,
        std_dev_acc=std_dev_acc,
        fig=fig,
        tested_parameter_name="gap_vs_noise",
        tested_parameter_values={"gaps": GAP_NOISE_GAPS, "noise_counts": GAP_NOISE_COUNTS},
        loss_fn_class=loss_fn_class,
        output_dir=output_dir
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

# train and test conditions

def robustness_train_test_gaps(model_class, train_gaps=GAPS_TRAIN, test_gaps=GAPS_TEST,
                               optimizer_class=config.OPTIMIZER, optimizer_params=config.OPTIMIZER_PARAMS,
                               loss_fn_class=config.LOSS_FN, scheduler_class=config.SCHEDULER,
                               scheduler_params=config.SCHEDULER_PARAMS, seed=config.SEED,
                               train_noise=True, test_noise=False, output_dir=None, show_plot=False):
    """
    Evaluates cross-gap transfer and temporal memory persistence by training independent models
    across varying train gap lengths (optionally noise-aware) and benchmarking each frozen model
    across all test gap lengths.

    For each training gap in `train_gaps`, a model is instantiated, initialized via fluctuation-driven
    scaling, and trained for `config.EPOCHS`. The frozen model is then evaluated across all gaps in
    `test_gaps` over multiple stochastic test datasets (`N_ROBUSTNESS_DATASET`).
    Renders comparative curves and a 2D Transfer Heatmap (Train GAP vs Test GAP), automatically archiving
    all CSV metrics, PNG figures, and metadata under 'robustness_results/different_gaps_train_test/<timestamp>/'.

    Args:
        model_class (type): SNN architecture class to train (e.g., classes.XORSNN).
        train_gaps (list[int]): Temporal gap durations used during model training.
        test_gaps (list[int]): Temporal gap durations evaluated during frozen inference.
        optimizer_class (type): Optimizer class used for training.
        optimizer_params (dict): Parameters passed to the optimizer constructor.
        loss_fn_class (type): Loss function criterion class.
        scheduler_class (type): Learning-rate scheduler class.
        scheduler_params (dict): Parameters passed to the scheduler constructor.
        seed (int): Deterministic RNG seed for model initialization and PRNG baseline.
        train_noise (bool): If True, trains with multi-condition mixture noise; clean otherwise.
        test_noise (bool): If True, tests with multi-condition mixture noise; clean otherwise.
        output_dir (str, optional): Target directory to save artifacts. If None, uses default timestamped path.
        show_plot (bool, optional): If True, opens interactive matplotlib window; closes figures otherwise.

    Returns:
        tuple:
            avg_loss (float): Mean test loss computed across all (train_gap, test_gap) pairs.
            avg_acc (float): Mean test accuracy computed across all (train_gap, test_gap) pairs.
            std_dev_loss (float): Standard deviation of test loss across all pairs.
            std_dev_acc (float): Standard deviation of test accuracy across all pairs.
    """
    acc_matrix = torch.zeros((len(train_gaps), len(test_gaps)))                         # rows: train, cols: test
    loss_matrix = torch.zeros((len(train_gaps), len(test_gaps)))
    summary = []

    for i, tr_gap in enumerate(train_gaps):
        print(f"\n[{i+1}/{len(train_gaps)}] Training model on GAP = {tr_gap} (noise={train_noise})...")
        torch.manual_seed(seed)
        model = model_class().to(config.DEVICE)

        train_dataset, val_dataset = functions.generate_train_validation_dataset(gap=tr_gap, seed=seed, noise=train_noise)
        train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)

        functions.fluctuation_driven_initialization(model, train_loader)                 # physics-grounded scaling
        loss_fn = loss_fn_class()
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = scheduler_class(optimizer, **scheduler_params)

        for ep in range(config.EPOCHS):
            functions.train(model, train_loader, optimizer, loss_fn)
            scheduler.step()

        print(f"  Evaluating frozen model (trained on GAP={tr_gap}) across {len(test_gaps)} test gaps...")
        model.eval()
        for j, te_gap in enumerate(test_gaps):
            te_losses, te_accs = [], []
            for N in range(config.N_ROBUSTNESS_DATASET):
                te_dataset = functions.generate_test_dataset(gap=te_gap, seed=seed + N, noise=test_noise)
                te_loader = functions.generate_test_dataloader(te_dataset)
                l, a = functions.evaluate(model, te_loader, loss_fn)
                te_losses.append(l)
                te_accs.append(a)

            m_loss, s_loss = sum(te_losses) / len(te_losses), torch.tensor(te_losses).std().item()
            m_acc, s_acc = sum(te_accs) / len(te_accs), torch.tensor(te_accs).std().item()
            acc_matrix[i, j] = m_acc
            loss_matrix[i, j] = m_loss

            summary.append({
                "train_gap": tr_gap,
                "test_gap": te_gap,
                "mean_test_loss": m_loss,
                "std_test_loss": s_loss,
                "mean_test_accuracy": m_acc,
                "std_test_accuracy": s_acc,
                "min_test_accuracy": min(te_accs),
                "max_test_accuracy": max(te_accs),
                "train_noise": train_noise,
                "test_noise": test_noise
            })

    fig_curves, ax_curves = plt.subplots(1, 2, figsize=(14, 5))
    for i, tr_gap in enumerate(train_gaps):
        ax_curves[0].plot(test_gaps, acc_matrix[i].numpy(), marker="o", label=f"Train GAP = {tr_gap}")
        ax_curves[1].plot(test_gaps, loss_matrix[i].numpy(), marker="o", label=f"Train GAP = {tr_gap}")
    ax_curves[0].set_title("Test Accuracy vs Test GAP")
    ax_curves[0].set_xlabel("Test GAP (timesteps)")
    ax_curves[0].set_ylabel("Accuracy (%)")
    ax_curves[0].grid(True, linestyle="--", alpha=0.5)
    ax_curves[0].legend()
    ax_curves[1].set_title("Test Loss vs Test GAP")
    ax_curves[1].set_xlabel("Test GAP (timesteps)")
    ax_curves[1].set_ylabel("Loss")
    ax_curves[1].grid(True, linestyle="--", alpha=0.5)
    ax_curves[1].legend()
    fig_curves.suptitle("Cross-GAP Generalization Curves")
    fig_curves.tight_layout()

    fig_map, ax_map = plt.subplots(figsize=(10, 6))
    im = ax_map.imshow(acc_matrix.numpy(), cmap="viridis", aspect="auto", vmin=50, vmax=100)
    ax_map.set_xticks(range(len(test_gaps)))
    ax_map.set_xticklabels(test_gaps)
    ax_map.set_yticks(range(len(train_gaps)))
    ax_map.set_yticklabels(train_gaps)
    ax_map.set_xlabel("Test GAP (timesteps)")
    ax_map.set_ylabel("Train GAP (timesteps)")
    ax_map.set_title("Cross-GAP Transfer Accuracy Heatmap (%)")
    for i in range(len(train_gaps)):
        for j in range(len(test_gaps)):
            val = acc_matrix[i, j].item()
            color = "black" if val > 80 else "white"
            ax_map.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax_map, label="Test Accuracy (%)")
    fig_map.tight_layout()

    avg_loss = loss_matrix.mean().item()
    avg_acc = acc_matrix.mean().item()
    std_dev_loss = loss_matrix.std().item()
    std_dev_acc = acc_matrix.std().item()

    save_robustness_results(
        test_name="different_gaps_train_test",
        model_class=model_class,
        results={"losses": loss_matrix, "accuracies": acc_matrix},
        summary=summary,
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        std_dev_loss=std_dev_loss,
        std_dev_acc=std_dev_acc,
        fig=fig_curves,
        summary_fig=fig_map,
        tested_parameter_name="train_test_gaps",
        tested_parameter_values={"train_gaps": train_gaps, "test_gaps": test_gaps},
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        loss_fn_class=loss_fn_class,
        scheduler_class=scheduler_class,
        scheduler_params=scheduler_params,
        output_dir=output_dir
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig_curves)
        if fig_map is not None:
            plt.close(fig_map)
    return avg_loss, avg_acc, std_dev_loss, std_dev_acc

# saving functions

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
    scheduler_params=config.SCHEDULER_PARAMS,
    output_dir=None
):
    """
    Stores the results and configuration of a robustness test in an
    archive directory.

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
        output_dir (str | None): Directory where results are archived.

    Returns:
        str: Path of the directory where results have been stored.
    """

    # create directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if output_dir is not None:
        archive_dir = output_dir
    else:
        archive_dir = os.path.join(
            "robustness",
            "results",
            test_name,
            timestamp
        )
    os.makedirs(archive_dir, exist_ok=True)

    # metadata we want to save
    metadata = {
        "test_name": test_name,
        "date": timestamp,

        "model": model_class.__name__ if hasattr(model_class, "__name__") else str(model_class),
        "device": str(config.DEVICE),

        "epochs": config.EPOCHS,
        "hidden_layer": config.HIDDEN_LAYER,
        "beta": config.BETA,

        "optimizer": optimizer_class.__name__ if hasattr(optimizer_class, "__name__") else str(optimizer_class),
        "optimizer_params": optimizer_params,

        "scheduler": scheduler_class.__name__ if hasattr(scheduler_class, "__name__") else str(scheduler_class),
        "scheduler_params": scheduler_params,

        "loss_fn": loss_fn_class.__name__ if hasattr(loss_fn_class, "__name__") else str(loss_fn_class),

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


## --- CONFIG & MODEL LOADING HELPERS ---

def load_robustness_config(config_name_or_path):
    """
    Loads robustness study configuration JSON from robustness/inputs/ or custom path.

    Args:
        config_name_or_path (str): File name (with or without .json) located in
            'robustness/inputs/' or an explicit filesystem path.

    Returns:
        dict: Parsed study configuration dictionary containing study name, models list,
            and active test execution flags.

    Raises:
        FileNotFoundError: If the specified config file does not exist in the filesystem.
    """
    if not os.path.exists(config_name_or_path):                               # check if direct path exists
        candidate = os.path.join("robustness", "inputs", config_name_or_path) # search in robustness/inputs/
        if os.path.exists(candidate):
            config_name_or_path = candidate
        elif not config_name_or_path.endswith(".json"):                       # check with .json suffix
            candidate = os.path.join("robustness", "inputs", f"{config_name_or_path}.json")
            if os.path.exists(candidate):
                config_name_or_path = candidate
    if not os.path.exists(config_name_or_path):                               # verify resolved candidate path
        raise FileNotFoundError(f"Robustness config not found: {config_name_or_path}")
    with open(config_name_or_path, "r") as file:                              # parse configuration file
        return json.load(file)

def load_model_from_config(model_cfg):
    """
    Instantiates and loads model from a checkpoint (.pt or folder/pth) specified in model_cfg.

    Inspects checkpoint metadata for architectural hyperparameters (hidden_layer, beta,
    learn_beta, reset_mechanism). Falls back to model_cfg values or default config settings.

    Args:
        model_cfg (dict): Model configuration dictionary containing checkpoint path
            and optional architecture parameters.

    Returns:
        torch.nn.Module: Loaded and evaluated XORSNN model on config.DEVICE.
    """
    ckpt_path = model_cfg.get("checkpoint_path", "")                          # target checkpoint path
    hidden = model_cfg.get("hidden_layer", config.HIDDEN_LAYER)               # hidden layer dimension
    beta = model_cfg.get("beta", config.BETA)                                 # membrane decay factor
    learn_beta = model_cfg.get("learn_beta", False)                           # trainable beta flag
    reset_mechanism = model_cfg.get("reset_mechanism", "zero")                # neuron reset mechanism

    # Resolve folder format from models/
    if os.path.isdir(ckpt_path):                                              # folder checkpoint format
        folder_name = os.path.basename(ckpt_path.rstrip("/\\"))
        pth_path = os.path.join(ckpt_path, f"{folder_name}.pth")
        if os.path.exists(pth_path):
            ckpt_path = pth_path

    if os.path.isfile(ckpt_path):                                             # load existing checkpoint
        raw = torch.load(ckpt_path, map_location=config.DEVICE)
        if isinstance(raw, dict) and "model_state_dict" in raw:               # Optuna / HPO checkpoint structure
            params = raw.get("params", {})
            hidden = params.get("hidden_layer", hidden)
            beta = params.get("beta", beta)
            learn_beta = params.get("learn_beta", learn_beta)
            reset_mechanism = params.get("reset_mechanism", reset_mechanism)
            state_dict = raw["model_state_dict"]
        elif isinstance(raw, dict) and any(k.startswith("net.") for k in raw.keys()):
            state_dict = raw                                                  # raw state dict format
        else:
            state_dict = raw

        model = classes.XORSNN(                                               # instantiate matching network
            hidden_layer=int(hidden),
            beta=float(beta) if not isinstance(beta, torch.Tensor) else beta,
            learn_beta=bool(learn_beta),
            reset_mechanism=str(reset_mechanism)
        ).to(config.DEVICE)
        model.load_state_dict(state_dict)                                     # restore synaptic weights
        model.eval()                                                          # freeze for inference
        beta_str = f"{beta:.4f}" if isinstance(beta, float) else "[heterogeneous]"
        print(f"Model loaded: {ckpt_path} (hidden={hidden}, beta={beta_str}, learn_beta={learn_beta}, reset={reset_mechanism})")
        return model
    else:
        print(f"Warning: Checkpoint '{ckpt_path}' not found. Falling back to functions.load_model().")
        return functions.load_model(hidden=hidden, beta=beta)                 # fallback instantiation


## --- MAIN CODE ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Config-driven SNN Robustness Evaluation")
    parser.add_argument("--config", type=str, default="test_4_upper_bound.json",
                        help="Robustness config JSON file in robustness/inputs/ or custom path")
    parser.add_argument("--show-plots", action="store_true", default=False,
                        help="Display GUI plots with plt.show() during evaluation")
    args = parser.parse_args()

    cfg = load_robustness_config(args.config)                                 # parse study configuration
    study_name = cfg.get("study_name", "robustness_study")                    # study directory identifier
    models_list = cfg.get("models", [])                                       # target models to benchmark
    tests_cfg = cfg.get("tests", {})                                          # boolean test selection flags
    show_plots = args.show_plots                                              # interactive display toggle

    print(f"\n=======================================================")
    print(f"Starting Robustness Suite: {study_name}")
    print(f"Description: {cfg.get('description', '')}")
    print(f"Total models to evaluate: {len(models_list)}")
    print(f"=======================================================\n")

    for i, model_cfg in enumerate(models_list):
        model_name = model_cfg.get("name", f"model_{i}")
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f">>> Evaluating Model [{i+1}/{len(models_list)}]: {model_name}")
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

        model = load_model_from_config(model_cfg)                             # restore model checkpoint
        model_dir = os.path.join("robustness", "results", study_name, model_name)
        os.makedirs(model_dir, exist_ok=True)                                 # create per-model output directory

        if tests_cfg.get("seeds_train", False):
            print(f"\n--- Running Seeds Training Robustness for {model_name} ---")
            train_dataset, val_dataset = functions.generate_train_validation_dataset()
            train_loader, val_loader = functions.generate_train_validation_dataloader(train_dataset, val_dataset)
            robustness_train_seeds(classes.XORSNN, train_loader, val_loader, output_dir=os.path.join(model_dir, "different_seeds_train"), show_plot=show_plots)

        if tests_cfg.get("gaps_train", False):
            print(f"\n--- Running Gaps Training Robustness for {model_name} ---")
            robustness_train_gaps(classes.XORSNN, noise=False, output_dir=os.path.join(model_dir, "different_gaps_train"), show_plot=show_plots)

        if tests_cfg.get("gaps_test_clean", False):
            print(f"\n--- Running Frozen Gaps Test (Clean) for {model_name} ---")
            robustness_test_gaps(model, noise=False, output_dir=os.path.join(model_dir, "different_gaps_test_clean"), show_plot=show_plots)

        if tests_cfg.get("gaps_test_noise", False):
            print(f"\n--- Running Frozen Gaps Test (With Noise) for {model_name} ---")
            robustness_test_gaps(model, noise=True, output_dir=os.path.join(model_dir, "different_gaps_test_noise"), show_plot=show_plots)

        if tests_cfg.get("noise_test", False):
            print(f"\n--- Running Noise Channel Sensitivity Test for {model_name} ---")
            robustness_test_noise(model, output_dir=os.path.join(model_dir, "different_noise_test"), show_plot=show_plots)

        if tests_cfg.get("probabilities_test", False):
            print(f"\n--- Running Firing Probabilities Sensitivity Test for {model_name} ---")
            robustness_test_probabilities(model, output_dir=os.path.join(model_dir, "different_probabilities_test"), show_plot=show_plots)

        if tests_cfg.get("gap_noise_3d_test", False):
            print(f"\n--- Running 3D Gap vs Noise Test for {model_name} ---")
            robustness_test_gap_noise(model, output_dir=os.path.join(model_dir, "different_gap_noise_test"), show_plot=show_plots)

    print(f"\n=======================================================")
    print(f"Robustness Suite '{study_name}' Completed Successfully!")
    print(f"Results archived under: robustness/results/{study_name}/")
    print(f"=======================================================\n")

