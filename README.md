# ML In C: Neural Networks from Scratch in C

A simple implementation of neural network concepts built entirely from scratch in C, without relying on external machine learning libraries. This project includes both a linear regression/perceptron example and a small neural network implementation of the AND logic gate.

## Features

- AND Gate Neural Network: Demonstrates a small neural network that learns the AND logic function.
- Linear Regression Model: Implements a basic perceptron-like model for linear regression (y = wx + b).
- Mean Squared Error Loss: Calculates loss using MSE for regression tasks.
- Numerical Gradient Descent: Uses finite differences to approximate gradients (no automatic differentiation).
- Training Loop: Iterative optimization with configurable learning rate, step size, and epochs.

## Prerequisites

- **Compiler**: GCC or any C compiler (tested with GCC on Windows via MSYS2).

## Building and Running

1. **Compile the perceptron regression example**:
   ```
   gcc perceptron.c -o perceptron.exe
   ```

2. **Run the perceptron example**:
   ```
   ./perceptron.exe
   ```

   This program initializes random weights and bias, computes initial loss, trains the model for 2500 epochs using gradient descent, and prints loss after each epoch and final weights.

3. **Compile the AND gate example**:
   ```
   gcc andGate.c -o andGate.exe
   ```

4. **Run the AND gate example**:
   ```
   ./andGate.exe
   ```

   This program trains a small neural network to learn the AND logic function from truth table examples.

## Code Structure

- `MSLoss()`: Computes mean squared error for the dataset.
- `train()`: Performs gradient descent training loop.
- `main()`: Sets up training data, initializes parameters, and runs training.
