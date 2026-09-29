# Temporal XOR with Spiking Neural Networks

This repository contains the source code for my Master's thesis in Engineering Physics at Politecnico di Milano. 

## Project Overview
The project investigates the training of Spiking Neural Networks (SNNs) to solve a Temporal XOR logic task. The models process rate-coded inputs, translating signal intensity into spike frequencies. The primary objective is to evaluate and compare different network architectures—specifically FeedForward and Reservoir computing—using the `snnTorch` and `PyTorch` frameworks.

### Key Features
* **Rate Coding Translation:** Conversion of standard input arrays into temporal spike trains.
* **Architecture Comparison:** Implementation of Leaky Integrate-and-Fire (LIF) neurons in both standard FeedForward and Reservoir configurations.
* **Best Model Selection:** Automated callback logic to evaluate test set accuracy during training, saving model weights (`.pth`) exclusively at peak performance to prevent overfitting.

## Repository Structure
* `main.py`: The entry point of the pipeline. It handles the training loop, validation tracking, and final robustness testing.
* `config.py`: Contains global hyperparameters and training configurations (e.g., `EPOCHS`, `LEARNING_RATE`).
* `classes.py`: Defines the SNN architectures, neuron parameters, and custom dataset classes.
* `functions.py`: Contains modular helper functions for the training/testing steps and model weight management.

## Hardware & Dependencies
This project is built using **Python 3** and relies heavily on `torch` and `snntorch`. 

For optimal performance, the training pipeline is configured for GPU acceleration via CUDA. The network simulations and tensor operations have been tested and optimized for execution on an NVIDIA GeForce RTX 5080.

## Usage
To start the training and evaluation process, clone the repository and execute the main script:

```bash
python main.py
