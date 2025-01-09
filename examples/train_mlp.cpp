#include "dl/nn/linear.hpp"
#include "dl/ops/activations.hpp"
#include "dl/ops/loss.hpp"
#include "dl/optim/optimizer.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// Simple 3-layer MLP for binary classification
class MLP {
public:
    MLP()
        : layer1(2, 2),  // 2 -> 2
          layer2(2, 1),  // 2 -> 1
          h1_(std::vector<size_t>{1, 2}),
          h1_act_(std::vector<size_t>{1, 2}),
          out_(std::vector<size_t>{1, 1}),
          final_(std::vector<size_t>{1, 1})
    {
        // For XOR, 2->2->1 is sufficient with non-linear activation
        // Initialize weights with Xavier/Glorot initialization
        float scale1 = std::sqrt(2.0f / (2 + 2)); // sqrt(2 / (in_features + out_features))
        float scale2 = std::sqrt(2.0f / (2 + 1));
        
        layer1.init_weights(scale1);
        layer2.init_weights(scale2);

        // Mark internal tensors as requiring grad if you want to watch them in debug
        h1_.set_requires_grad(true);
        h1_.grad().resize(h1_.data().size(), 0.0f);
        h1_act_.set_requires_grad(true);
        h1_act_.grad().resize(h1_act_.data().size(), 0.0f);
        out_.set_requires_grad(true);
        out_.grad().resize(out_.data().size(), 0.0f);
        final_.set_requires_grad(true);
        final_.grad().resize(final_.data().size(), 0.0f);
    }

    dl::Tensor<float> forward(const dl::Tensor<float>& x)
    {
        // First layer
        h1_ = layer1.forward(x);  
        h1_.set_requires_grad(true);
        h1_.grad().resize(h1_.data().size(), 0.0f);

        // ReLU activation
        h1_act_ = dl::ops::relu(h1_);
        h1_act_.set_requires_grad(true);
        h1_act_.grad().resize(h1_act_.data().size(), 0.0f);

        // Second layer (output)
        out_ = layer2.forward(h1_act_);
        out_.set_requires_grad(true);
        out_.grad().resize(out_.data().size(), 0.0f);

        // Sigmoid activation to produce final probability in [0,1]
        final_ = dl::ops::sigmoid(out_);
        final_.set_requires_grad(true);
        final_.grad().resize(final_.data().size(), 0.0f);

        // Return a copy to avoid modifying the internal state
        return final_;
    }

    // The layers
    dl::nn::Linear<float> layer1;
    dl::nn::Linear<float> layer2;

private:
    // Minimal internal tensors for storing intermediate outputs
    dl::Tensor<float> h1_;      // Hidden layer output
    dl::Tensor<float> h1_act_;  // Hidden activation
    dl::Tensor<float> out_;     // Raw output of second layer
    dl::Tensor<float> final_;   // Final (sigmoid) output
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
    dl::optim::SGD<float> optimizer(0.01f);  // Increased learning rate
    
    // Add model parameters to optimizer
    model.layer1.add_parameters_to_optimizer(optimizer);
    model.layer2.add_parameters_to_optimizer(optimizer);
    
    // Generate training data
    std::vector<dl::Tensor<float>> inputs;
    std::vector<dl::Tensor<float>> targets;
    generate_data(inputs, targets, 2000);  // More training data
    
    // Training loop
    float best_accuracy = 0.0f;
    float prev_loss = std::numeric_limits<float>::max();
    int patience = 0;
    const int max_patience = 20;
    const size_t batch_size = 32;
    const int max_epochs = 100;  // Increased epochs
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;
        
        // Shuffle the data
        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Process mini-batches
        for (size_t batch_start = 0; batch_start < inputs.size(); batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, inputs.size());
            float batch_size_f = static_cast<float>(batch_end - batch_start);
            
            // Zero gradients before forward pass
            optimizer.zero_grad();
            
            // Clear computation graph before forward pass
            dl::ComputationGraph::getInstance().clear();
            
            // Accumulate gradients over batch
            float batch_loss = 0.0f;
            int batch_correct = 0;
            
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];
                
                // Forward pass
                dl::Tensor<float> pred = model.forward(inputs[idx]);
                
                // Compute loss for this batch
                dl::Tensor<float> loss = dl::ops::mse_loss(pred, targets[idx]);
                
                // Scale the loss gradient by batch size
                loss.grad().clear();
                loss.grad().resize(1, 1.0f / batch_size_f);
                
                // Backward pass
                dl::ComputationGraph::getInstance().backward();
                
                // Track accuracy
                bool correct_prediction = (pred.data()[0] >= 0.5f) == (targets[idx].data()[0] >= 0.5f);
                if (correct_prediction) {
                    batch_correct++;
                }
                
                batch_loss += loss.data()[0];
            }
            
            // Update weights
            optimizer.step();
            
            // Print weights after update for first batch
            if (epoch == 0 && batch_start == 0) {
                std::cout << "\nLayer weights after optimizer.step():" << std::endl;
                for (size_t j = 0; j < 3; ++j) {
                    std::cout << "Layer 1 weight[" << j << "]: " << model.layer1.weights().data()[j] << std::endl;
                    std::cout << "Layer 2 weight[" << j << "]: " << model.layer2.weights().data()[j] << std::endl;
                }
                
                std::cout << "\nGradients after optimizer updated the weights:" << std::endl;
                for (size_t j = 0; j < 3; ++j) {
                    std::cout << "Layer 1 grad[" << j << "]: " << model.layer1.weights().grad()[j] << std::endl;
                    std::cout << "Layer 2 grad[" << j << "]: " << model.layer2.weights().grad()[j] << std::endl;
                }
            }
            
            // Print gradients after zero_grad for first batch
            if (epoch == 0 && batch_start == 0) {
                std::cout << "\nGradients after optimizer.zero_grad():" << std::endl;
                for (size_t j = 0; j < 3; ++j) {
                    std::cout << "Layer 1 grad[" << j << "]: " << model.layer1.weights().grad()[j] << std::endl;
                    std::cout << "Layer 2 grad[" << j << "]: " << model.layer2.weights().grad()[j] << std::endl;
                }
            }
            
            // Track metrics
            total_loss += batch_loss;
            correct += batch_correct;
        }
        
        // Calculate metrics
        float avg_loss = total_loss / inputs.size();
        float accuracy = 100.0f * correct / inputs.size();
        best_accuracy = std::max(best_accuracy, accuracy);
        
        // Print progress every epoch
        std::cout << "Epoch " << epoch + 1 << "/" << 100
                 << " - Loss: " << avg_loss 
                 << " - Accuracy: " << accuracy << "% "
                 << " - Best: " << best_accuracy << "%" << std::endl;
        
        // Early stopping check
        if (avg_loss >= prev_loss) {
            patience++;
            if (patience >= max_patience) {
                std::cout << "\nEarly stopping triggered after " << epoch + 1 << " epochs" << std::endl;
                break;
            }
        } else {
            patience = 0;
            prev_loss = avg_loss;
        }
    }
    
    return 0;
}
