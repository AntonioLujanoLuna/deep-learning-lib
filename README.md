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
#include <iostream>
#include <vector>
#include <random>

// Generate simple XOR dataset
void generate_data(std::vector<dl::Tensor<float>>& inputs, 
                  std::vector<dl::Tensor<float>>& targets, 
                  size_t n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < n_samples; ++i) {
        // Create input tensor: shape [1, 2]
        dl::Tensor<float> input({1, 2});
        input.set_requires_grad(true);
        input.data()[0] = dist(gen);
        input.data()[1] = dist(gen);
        
        // Create target tensor: shape [1, 1]
        dl::Tensor<float> target({1, 1});
        // XOR-like pattern: true if exactly one input is > 0.5
        target.data()[0] = (input.data()[0] > 0.5f) != (input.data()[1] > 0.5f) ? 1.0f : 0.0f;
        
        inputs.push_back(input);
        targets.push_back(target);
    }
}

int main() {
    // Create model: 2 inputs -> 8 hidden -> 8 hidden -> 1 output
    dl::nn::MLP<float> model({2, 8, 8, 1});
    float learning_rate = 0.01f;

    // Generate training data: 100 samples
    std::vector<dl::Tensor<float>> inputs;
    std::vector<dl::Tensor<float>> targets;
    generate_data(inputs, targets, 100);
    
    // Training parameters
    const int epochs = 1000;
    float best_accuracy = 0.0f;
    
    std::cout << "Starting training..." << std::endl;
    std::cout << "Model has " << model.num_parameters() << " parameters" << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;
        
        // Train on each sample
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Clear gradients and computation graph
            model.zero_grad();
            dl::ComputationGraph::getInstance().clear();
            
            // Forward pass
            auto pred = model.forward(inputs[i]);
            auto loss = dl::ops::binary_cross_entropy(pred, targets[i]);
            
            // Backward pass
            loss->grad().assign(1, 1.0f);
            dl::ComputationGraph::getInstance().backward();
            
            // Update parameters
            model.update_parameters(learning_rate);
            
            // Track metrics
            total_loss += loss->data()[0];
            bool correct_prediction = (pred->data()[0] >= 0.5f) == (targets[i].data()[0] >= 0.5f);
            if (correct_prediction) correct++;
        }
        
        // Calculate metrics
        float avg_loss = total_loss / inputs.size();
        float accuracy = 100.0f * correct / inputs.size();
        best_accuracy = std::max(best_accuracy, accuracy);
        
        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                     << " - Loss: " << avg_loss 
                     << " - Accuracy: " << accuracy << "% "
                     << " - Best: " << best_accuracy << "%" << std::endl;
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Best accuracy achieved: " << best_accuracy << "%" << std::endl;
    
    return 0;
}
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