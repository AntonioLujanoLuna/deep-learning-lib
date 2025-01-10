#include "dl/tensor.hpp"
#include "dl/nn/linear.hpp"
#include "dl/ops/loss.hpp"
#include "dl/ops/activations.hpp"
#include "dl/ops/matrix_ops.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <iomanip>

// Simple 3-layer MLP for binary classification
class MLP {
public:
    MLP(size_t input_dim, size_t hidden_dim, size_t output_dim) 
        : layer1(std::make_shared<dl::nn::Linear<float>>(input_dim, hidden_dim))
        , layer2(std::make_shared<dl::nn::Linear<float>>(hidden_dim, output_dim)) {
        
        // Ensure all parameters require gradients
        layer1->weights().set_requires_grad(true);
        layer1->bias().set_requires_grad(true);
        layer2->weights().set_requires_grad(true);
        layer2->bias().set_requires_grad(true);
    }

    std::shared_ptr<dl::Tensor<float>> forward(const dl::Tensor<float>& x) {
        std::cout << "\n=== MLP Forward Pass ===" << std::endl;
        std::cout << "Input dimensions: [" << x.shape()[0] << ", " << x.shape()[1] << "]" << std::endl;
        
        // First layer
        auto h1 = layer1->forward(x);  
        std::cout << "h1 dimensions: [" << h1->shape()[0] << ", " << h1->shape()[1] << "]" << std::endl;

        // ReLU activation
        auto h1_act = dl::ops::relu(*h1);
        std::cout << "h1_act dimensions: [" << h1_act->shape()[0] << ", " << h1_act->shape()[1] << "]" << std::endl;

        // Second layer (output)
        auto out = layer2->forward(*h1_act);
        std::cout << "out dimensions: [" << out->shape()[0] << ", " << out->shape()[1] << "]" << std::endl;

        // Sigmoid activation to produce final probability in [0,1]
        auto final = dl::ops::sigmoid(*out);  
        std::cout << "final dimensions: [" << final->shape()[0] << ", " << final->shape()[1] << "]" << std::endl;

        // Print first few values
        std::cout << "First few output values: ";
        for (size_t i = 0; i < std::min(size_t(3), final->data().size()); ++i) {
            std::cout << final->data()[i] << " ";
        }
        std::cout << std::endl;
        
        return final;
    }

    void zero_grad() {
        layer1->zero_grad();
        layer2->zero_grad();
    }

    std::vector<float> parameters() const {
        std::vector<float> params;
        
        // Get parameters from layer1
        const auto& w1 = layer1->weights().data();
        const auto& b1 = layer1->bias().data();
        params.insert(params.end(), w1.begin(), w1.end());
        params.insert(params.end(), b1.begin(), b1.end());
        
        // Get parameters from layer2
        const auto& w2 = layer2->weights().data();
        const auto& b2 = layer2->bias().data();
        params.insert(params.end(), w2.begin(), w2.end());
        params.insert(params.end(), b2.begin(), b2.end());
        
        return params;
    }

    void update_parameters(float learning_rate) {
        // Update layer1 parameters
        auto& w1 = layer1->weights();
        auto& b1 = layer1->bias();
        const auto& w1_grad = w1.grad();
        const auto& b1_grad = b1.grad();
        
        for (size_t i = 0; i < w1.data().size(); ++i) {
            w1.data()[i] -= learning_rate * w1_grad[i];
        }
        for (size_t i = 0; i < b1.data().size(); ++i) {
            b1.data()[i] -= learning_rate * b1_grad[i];
        }
        
        // Update layer2 parameters
        auto& w2 = layer2->weights();
        auto& b2 = layer2->bias();
        const auto& w2_grad = w2.grad();
        const auto& b2_grad = b2.grad();
        
        for (size_t i = 0; i < w2.data().size(); ++i) {
            w2.data()[i] -= learning_rate * w2_grad[i];
        }
        for (size_t i = 0; i < b2.data().size(); ++i) {
            b2.data()[i] -= learning_rate * b2_grad[i];
        }
    }
    
    // Getter methods for debugging
    const std::shared_ptr<dl::nn::Linear<float>>& get_layer1() const { return layer1; }
    const std::shared_ptr<dl::nn::Linear<float>>& get_layer2() const { return layer2; }

private:
    std::shared_ptr<dl::nn::Linear<float>> layer1;
    std::shared_ptr<dl::nn::Linear<float>> layer2;
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
    std::cout << "Starting MLP training example..." << std::endl;

    // Create model and optimizer
    MLP model(2, 2, 1);
    std::cout << "Created MLP" << std::endl;
    float learning_rate = 0.1f;  // Increased learning rate for faster convergence
    std::cout << "Created optimizer" << std::endl;

    // Generate training data
    std::vector<dl::Tensor<float>> inputs;
    std::vector<dl::Tensor<float>> targets;
    generate_data(inputs, targets, 1000);  // Reduced data size for faster training
    
    // Training loop
    float best_accuracy = 0.0f;
    float prev_loss = std::numeric_limits<float>::max();
    int patience = 0;
    const int max_patience = 10;  // Reduced patience for faster stopping
    const size_t batch_size = 16;  // Reduced batch size
    const int max_epochs = 50;  // Reduced epochs
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        std::cout << "\nEpoch " << epoch << std::endl;
        
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
            
            // Clear gradients and graph before new batch
            model.zero_grad();
            dl::ComputationGraph::getInstance().clear();
            
            // Create batch tensors
            dl::Tensor<float> batch_input({batch_size, 2});
            dl::Tensor<float> batch_target({batch_size, 1});
            
            // Ensure batch input requires gradients
            batch_input.set_requires_grad(true);
            
            std::cout << "\n=== Processing Batch ===" << std::endl;
            std::cout << "Batch input dimensions: [" << batch_input.shape()[0] << ", " << batch_input.shape()[1] << "]" << std::endl;
            std::cout << "Batch target dimensions: [" << batch_target.shape()[0] << ", " << batch_target.shape()[1] << "]" << std::endl;
            
            // Fill batch tensors
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = indices[batch_start + i];
                const auto& input_data = inputs[idx].data();
                const auto& target_data = targets[idx].data();
                
                // Copy input data
                batch_input.data()[i*2] = input_data[0];
                batch_input.data()[i*2 + 1] = input_data[1];
                
                // Copy target data
                batch_target.data()[i] = target_data[0];
                
                if (i < 3) {  // Print first few samples
                    std::cout << "Sample " << i << ": input=(" << input_data[0] << "," << input_data[1] 
                              << "), target=" << target_data[0] << std::endl;
                }
            }
            
            // Single forward pass for entire batch
            std::cout << "\nStarting forward pass..." << std::endl;
            auto pred = model.forward(batch_input);
            std::cout << "Forward pass completed. Pred dimensions: [" << pred->shape()[0] << ", " << pred->shape()[1] << "]" << std::endl;
            
            // Compute loss for entire batch
            std::cout << "\nComputing loss..." << std::endl;
            auto loss = dl::ops::binary_cross_entropy(pred, batch_target);
            std::cout << "Loss computed: " << loss->data()[0] << std::endl;
            
            // Initialize loss gradient to 1.0
            loss->grad().assign(1, 1.0f);
            std::cout << "\nSet loss gradient to 1.0" << std::endl;
            
            // Single backward pass for entire batch
            std::cout << "\nStarting backward pass..." << std::endl;
            dl::ComputationGraph::getInstance().backward();
            std::cout << "Backward pass completed" << std::endl;
            
            // Print gradients before optimizer step
            std::cout << "\nGradients before optimizer step:" << std::endl;
            for (size_t j = 0; j < std::min(size_t(3), model.get_layer1()->weights().grad().size()); ++j) {
                std::cout << "Layer 1 grad[" << j << "]: " << model.get_layer1()->weights().grad()[j] << std::endl;
            }
            for (size_t j = 0; j < std::min(size_t(3), model.get_layer2()->weights().grad().size()); ++j) {
                std::cout << "Layer 2 grad[" << j << "]: " << model.get_layer2()->weights().grad()[j] << std::endl;
            }
            
            // Update weights after batch
            std::cout << "\nTaking optimizer step..." << std::endl;
            model.update_parameters(learning_rate);
            std::cout << "Optimizer step completed" << std::endl;
            
            // Print weights after update
            std::cout << "\nWeights after optimizer step:" << std::endl;
            for (size_t j = 0; j < std::min(size_t(3), model.get_layer1()->weights().data().size()); ++j) {
                std::cout << "Layer 1 weight[" << j << "]: " << model.get_layer1()->weights().data()[j] << std::endl;
            }
            for (size_t j = 0; j < std::min(size_t(3), model.get_layer2()->weights().data().size()); ++j) {
                std::cout << "Layer 2 weight[" << j << "]: " << model.get_layer2()->weights().data()[j] << std::endl;
            }
            
            // Clear computation graph after batch is complete
            dl::ComputationGraph::getInstance().clear();
            
            // Track accuracy
            int batch_correct = 0;
            float batch_loss = loss->data()[0];
            
            for (size_t i = 0; i < batch_size; ++i) {
                bool correct_prediction = (pred->data()[i] >= 0.5f) == (batch_target.data()[i] >= 0.5f);
                if (correct_prediction) {
                    batch_correct++;
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
        std::cout << "Epoch " << epoch + 1 << "/" << 50
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
    
    std::cout << "Training completed successfully" << std::endl;
    return 0;
}
