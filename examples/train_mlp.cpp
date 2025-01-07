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
        // First layer
        h1_ = layer1.forward(x);
        
        if (dl::ComputationGraph::getInstance().getNodes().empty()) {
            std::cout << "Warning: No nodes in computation graph after layer1" << std::endl;
        }
        
        // ReLU activation
        h1_act_ = dl::ops::relu(h1_);
        
        if (dl::ComputationGraph::getInstance().getNodes().size() < 2) {
            std::cout << "Warning: Missing ReLU node in computation graph" << std::endl;
        }
        
        // Second layer
        out_ = layer2.forward(h1_act_);
        
        if (dl::ComputationGraph::getInstance().getNodes().size() < 3) {
            std::cout << "Warning: Missing layer2 node in computation graph" << std::endl;
        }
        
        // Sigmoid activation
        final_ = dl::ops::sigmoid(out_);
        
        if (dl::ComputationGraph::getInstance().getNodes().size() < 4) {
            std::cout << "Warning: Missing sigmoid node in computation graph" << std::endl;
        }
        
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
                  size_t n_samples) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    inputs.clear();
    targets.clear();
    
    for (size_t i = 0; i < n_samples; ++i) {
        float x1 = dis(gen);
        float x2 = dis(gen);
        
        dl::Tensor<float> input({1, 2});
        input.data()[0] = x1;
        input.data()[1] = x2;
        input.set_requires_grad(true);
        inputs.push_back(input);
        
        dl::Tensor<float> target({1, 1});
        target.data()[0] = (x1 * x2 > 0) ? 1.0f : 0.0f;  // XOR-like pattern
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
            dl::Tensor<float> loss = dl::ops::binary_cross_entropy(pred, targets[i]);
            
            if (epoch == 0 && i == 0) {
                std::cout << "Debug - Epoch 0, First sample:" << std::endl;
                std::cout << "Prediction: " << pred.data()[0] << std::endl;
                std::cout << "Target: " << targets[i].data()[0] << std::endl;
                std::cout << "Loss: " << loss.data()[0] << std::endl;
                std::cout << "Loss gradient: " << loss.grad()[0] << std::endl;
                std::cout << "Computation graph size: " << dl::ComputationGraph::getInstance().getNodes().size() << std::endl;
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
