# ML In C: Neural Networks from Scratch in C

A simple implementation of neural network concepts (starting with perceptron/linear regression) built entirely from scratch in C, without relying on external machine learning libraries. This project demonstrates fundamental concepts like gradient descent, loss functions, and training loops using basic C primitives.

## Features

- Linear Regression Model: Implements a basic perceptron-like model for linear regression (y = wx + b).
- Mean Squared Error Loss: Calculates loss using MSE for regression tasks.
- Numerical Gradient Descent: Uses finite differences to approximate gradients (no automatic differentiation).
- Training Loop: Iterative optimization with configurable learning rate, step size, and epochs.

## Prerequisites

- **Compiler**: GCC or any C compiler (tested with GCC on Windows via MSYS2).
- **Headers**: Requires `<stdio.h>`, `<stdlib.h>`, and `<time.h>` (standard C library).

## Building and Running

1. **Compile**:
   ```
   gcc perceptron.c -o perceptron.exe
   ```

2. **Run**:
   ```
   ./perceptron.exe
   ```

   The program will initialize random weights and bias, compute initial loss, train the model for 2500 epochs using gradient descent, and print loss after each epoch and final weights.

## Code Structure

- `MSLoss()`: Computes mean squared error for the dataset.
- `train()`: Performs gradient descent training loop.
- `main()`: Sets up training data, initializes parameters, and runs training.
