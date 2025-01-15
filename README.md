# Deep Learning Library (DL)

A lightweight C++ deep learning library implementing automatic differentiation and neural network components from scratch. This library provides fundamental building blocks for creating and training neural networks with a PyTorch-like API.

## Features

- Automatic differentiation (autograd) system with dynamic computational graphs
- N-dimensional tensor operations with broadcasting support
- Basic neural network layers:
  - Linear (Fully Connected) layers
  - Multi-Layer Perceptron (MLP)
- Activation functions:
  - ReLU
  - Sigmoid
  - Tanh
- Loss functions:
  - Mean Squared Error (MSE)
  - Binary Cross Entropy (BCE)
- Optimization:
  - Stochastic Gradient Descent (SGD) with gradient clipping
- Matrix operations:
  - Matrix multiplication
  - Convolution operations (Conv2D)

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dl.git
cd dl
```

2. Create and enter build directory:
```bash
mkdir build && cd build
```

3. Configure and build:
```bash
cmake ..
make
```

## Usage Example

Here's a simple example of creating and training a binary classification model:

```cpp
#include "dl/tensor.hpp"
#include "dl/nn/mlp.hpp"
#include "dl/ops/loss.hpp"

// Create a simple MLP model
dl::nn::MLP<float> model({2, 8, 8, 1});  // 2 inputs, 2 hidden layers, 1 output

// Create input and target tensors
dl::Tensor<float> input({1, 2});
dl::Tensor<float> target({1, 1});
input.set_requires_grad(true);

// Forward pass
auto output = model.forward(input);

// Compute loss
auto loss = dl::ops::binary_cross_entropy(output, target);

// Backward pass
model.zero_grad();
dl::ComputationGraph::getInstance().backward();

// Update parameters
model.update_parameters(0.001f);  // learning rate = 0.001
```

## Project Structure

```
include/dl/
├── autograd.hpp       - Automatic differentiation system
├── tensor.hpp        - Tensor class implementation
├── utils.hpp         - Utility functions
├── nn/
│   ├── linear.hpp    - Linear layer implementation
│   └── mlp.hpp       - Multi-layer perceptron
├── ops/
│   ├── activations.hpp - Activation functions
│   ├── basic_ops.hpp  - Basic tensor operations
│   ├── loss.hpp       - Loss functions
│   └── matrix_ops.hpp - Matrix operations
└── optim/
    └── optimizer.hpp  - Optimization algorithms
```

## Testing

The project includes comprehensive unit tests using the doctest framework. To run the tests:

```bash
cd build
make test
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Make sure to:

1. Follow the existing code style
2. Add tests for any new functionality
3. Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by PyTorch's design and API
- Uses doctest for unit testing