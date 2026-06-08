# ML In C: Neural Networks from Scratch in C

A simple implementation of neural network concepts built entirely from scratch in C, without relying on external machine learning libraries.

## Features

- Perceptron Regression: Fits a linear model `y = wx + b` using finite-difference gradient descent.
- AND Gate Neural Network: Trains a single-neuron model with sigmoid activation for the AND logic function.
- XOR Gate Neural Network: Trains a small 2-layer neural network that can learn the XOR function using finite-difference gradients.
- Mean Squared Error Loss: Uses MSE to measure prediction error for both regression and classification examples.
- Numerical Gradient Descent: Approximates gradients with finite differences rather than automatic differentiation.
- Matrix Utilities: Includes `nn.h` to train Multi-layer perceptrons (MLP)

## Prerequisites

- **Compiler**: GCC or any C compiler (tested with GCC on Windows via MSYS2).

## Building and Running

1. **Compile the perceptron regression example**:
   ```sh
   gcc perceptron.c -o perceptron.exe
   ```

2. **Run the perceptron example**:
   ```sh
   ./perceptron.exe
   ```

   This program initializes random weights and bias, computes initial loss, trains, and prints loss per epoch and final model parameters.

3. **Compile the AND gate example**:
   ```sh
   gcc andGate.c -o andGate.exe
   ```

4. **Run the AND gate example**:
   ```sh
   ./andGate.exe
   ```

   This program trains a sigmoid neuron for the AND truth table and prints the learned output for each input pair.

5. **Compile the XOR gate example**:
   ```sh
   gcc xorGate.c -o xorGate.exe
   ```

6. **Run the XOR gate example**:
   ```sh
   ./xorGate.exe
   ```

   This program trains a small two-layer neural network to learn the XOR truth table using finite-difference gradients.

## Project Files

- `perceptron.c`: Linear regression example with finite-difference gradient descent.
- `andGate.c`: Single-neuron logic gate example using sigmoid activation.
- `xorGate.c`: Multi-layer neural network example for XOR.
- `nn.h/nn.h`: Minimal matrix utility header for training Multi-layer perceptrons (MLP)
