#include "dl/nn/linear.hpp"
#include "dl/ops/activations.hpp"
#include "dl/ops/loss.hpp"
#include "dl/optim/optimizer.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// Simple 2-layer MLP for binary classification
class MLP {
public:
    MLP() : 
        layer1(2, 8),   // Input size: 2, Hidden size: 8
        layer2(8, 1),   // Hidden size: 8, Output size: 1
        h1_(std::vector<size_t>{1, 8}),
        h1_act_(std::vector<size_t>{1, 8}),
        out_(std::vector<size_t>{1, 1}),
        final_(std::vector<size_t>{1, 1})
    {
        h1_.set_requires_grad(true);
        h1_act_.set_requires_grad(true);
        out_.set_requires_grad(true);
        final_.set_requires_grad(true);
    }

    dl::Tensor<float>& forward(const dl::Tensor<float>& x) {
        std::cout << "\n=== Starting MLP forward pass ===" << std::endl;
        std::cout << "Input requires_grad: " << std::boolalpha << x.requires_grad() << std::endl;
        
        // First layer
        h1_ = layer1.forward(x);
        h1_.set_requires_grad(true);
        std::cout << "Layer1 output requires_grad: " << h1_.requires_grad() << std::endl;
        
        if (dl::ComputationGraph::getInstance().getNodes().empty()) {
            std::cout << "Warning: No nodes in computation graph after layer1" << std::endl;
        }
        
        // ReLU activation
        h1_act_ = dl::ops::relu(h1_);
        h1_act_.set_requires_grad(true);
        std::cout << "ReLU output requires_grad: " << h1_act_.requires_grad() << std::endl;
        
        if (dl::ComputationGraph::getInstance().getNodes().size() < 2) {
            std::cout << "Warning: Missing ReLU node in computation graph" << std::endl;
        }
        
        // Second layer
        out_ = layer2.forward(h1_act_);
        out_.set_requires_grad(true);
        std::cout << "Layer2 output requires_grad: " << out_.requires_grad() << std::endl;
        
        if (dl::ComputationGraph::getInstance().getNodes().size() < 3) {
            std::cout << "Warning: Missing layer2 node in computation graph" << std::endl;
        }
        
        // Sigmoid activation
        final_ = dl::ops::sigmoid(out_);
        final_.set_requires_grad(true);
        std::cout << "Final output requires_grad: " << final_.requires_grad() << std::endl;
        
        if (dl::ComputationGraph::getInstance().getNodes().size() < 4) {
            std::cout << "Warning: Missing sigmoid node in computation graph" << std::endl;
        }
        
        std::cout << "=== MLP forward pass completed ===" << std::endl;
        return final_;
    }

    void print_gradients() const {
        std::cout << "Layer 1 weights: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << layer1.weights().data()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Layer 1 weight grads: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << layer1.weights().grad()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Layer 2 weights: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << layer2.weights().data()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Layer 2 weight grads: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << layer2.weights().grad()[i] << " ";
        }
        std::cout << std::endl;
    }

    dl::nn::Linear<float> layer1;
    dl::nn::Linear<float> layer2;

private:
    dl::Tensor<float> h1_;      // First layer output
    dl::Tensor<float> h1_act_;  // First layer activation
    dl::Tensor<float> out_;     // Second layer output
    dl::Tensor<float> final_;   // Final output (after sigmoid)
};

// Generate a simple XOR-like dataset
void generate_data(std::vector<dl::Tensor<float>>& inputs, 
                  std::vector<dl::Tensor<float>>& targets, 
                  size_t n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < n_samples; ++i) {
        // Generate random input in [0,1] x [0,1]
        dl::Tensor<float> input({1, 2});
        input.set_requires_grad(true);  // Set requires_grad for input tensor
        input.data()[0] = dist(gen);
        input.data()[1] = dist(gen);
        
        // Target is 1 if points are in opposite quadrants (XOR-like)
        dl::Tensor<float> target({1, 1});
        target.set_requires_grad(false);  // Target should not require gradients
        target.data()[0] = (input.data()[0] > 0.5f) != (input.data()[1] > 0.5f) ? 1.0f : 0.0f;
        
        inputs.push_back(input);
        targets.push_back(target);
    }
}

int main() {
    // Create model and optimizer
    MLP model;
    dl::optim::SGD<float> optimizer(0.1f);  // Learning rate = 0.1
    
    // Add model parameters to optimizer
    model.layer1.add_parameters_to_optimizer(optimizer);
    model.layer2.add_parameters_to_optimizer(optimizer);
    
    // Generate training data
    std::vector<dl::Tensor<float>> inputs;
    std::vector<dl::Tensor<float>> targets;
    generate_data(inputs, targets, 100);  // 100 training samples
    
    std::cout << "Starting training for 1000 epochs..." << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Clear computation graph before forward pass
            dl::ComputationGraph::getInstance().clear();
            
            // Forward pass
            dl::Tensor<float>& pred = model.forward(inputs[i]);
            
            // Ensure prediction requires gradients
            if (!pred.requires_grad()) {
                pred.set_requires_grad(true);
            }
            
            dl::Tensor<float> loss = dl::ops::binary_cross_entropy(pred, targets[i]);
            
            if (epoch == 0 && i == 0) {
                std::cout << "Debug - Epoch 0, First sample:" << std::endl;
                std::cout << "Prediction: " << pred.data()[0] << std::endl;
                std::cout << "Target: " << targets[i].data()[0] << std::endl;
                std::cout << "Loss: " << loss.data()[0] << std::endl;
                std::cout << "Loss gradient: " << loss.grad()[0] << std::endl;
                std::cout << "Computation graph size: " << dl::ComputationGraph::getInstance().getNodes().size() << std::endl;
                
                // Print gradient states
                std::cout << "Gradients before backward:" << std::endl;
                model.print_gradients();
            }
            
            // Backward pass
            dl::ComputationGraph::getInstance().backward();
            
            // Update weights
            optimizer.step();
            optimizer.zero_grad();
            
            // Track metrics
            total_loss += loss.data()[0];
            if (std::abs(pred.data()[0] - targets[i].data()[0]) < 0.5f) {
                correct++;
            }
        }
        
        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            float avg_loss = total_loss / inputs.size();
            float accuracy = 100.0f * correct / inputs.size();
            std::cout << "Epoch " << epoch + 1 << " - ";
            std::cout << "Average Loss: " << avg_loss << ", ";
            std::cout << "Accuracy: " << accuracy << "%" << std::endl;
        }
    }
    
    return 0;
}
