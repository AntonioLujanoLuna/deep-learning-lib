#include "dl/tensor.hpp"
#include "dl/nn/mlp.hpp"
#include "dl/ops/loss.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <iomanip>
#include <numeric>

// Generate a simple XOR-like dataset
void generate_data(std::vector<dl::Tensor<float>>& inputs, 
                  std::vector<dl::Tensor<float>>& targets, 
                  size_t n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < n_samples; ++i) {
        dl::Tensor<float> input({1, 2});
        input.set_requires_grad(true);
        input.data()[0] = dist(gen);
        input.data()[1] = dist(gen);
        
        dl::Tensor<float> target({1, 1});
        target.set_requires_grad(false);
        target.data()[0] = (input.data()[0] > 0.5f) != (input.data()[1] > 0.5f) ? 1.0f : 0.0f;
        
        inputs.push_back(input);
        targets.push_back(target);
    }
}

int main() {
    std::cout << "Starting MLP training example..." << std::endl;

    // Create model with architecture [2, 4, 4, 1]
    dl::nn::MLP<float> model({2, 8, 8, 1});
    float learning_rate = 0.001f;

    // Generate training data
    std::vector<dl::Tensor<float>> inputs;
    std::vector<dl::Tensor<float>> targets;
    generate_data(inputs, targets, 1000);
    
    // Training loop parameters
    float best_accuracy = 0.0f;
    float prev_loss = std::numeric_limits<float>::max();
    int patience = 0;
    const int max_patience = 200;
    const size_t batch_size = 8;
    const int max_epochs = 10000;
    
    std::cout << "Model has " << model.num_parameters() << " parameters" << std::endl;
    
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
            size_t current_batch_size = batch_end - batch_start;
            
            model.zero_grad();
            dl::ComputationGraph::getInstance().clear();
            
            dl::Tensor<float> batch_input({current_batch_size, 2});
            dl::Tensor<float> batch_target({current_batch_size, 1});
            batch_input.set_requires_grad(true);
            
            // Fill batch tensors
            for (size_t i = 0; i < current_batch_size; ++i) {
                size_t idx = indices[batch_start + i];
                const auto& input_data = inputs[idx].data();
                const auto& target_data = targets[idx].data();
                batch_input.data()[i*2] = input_data[0];
                batch_input.data()[i*2 + 1] = input_data[1];
                batch_target.data()[i] = target_data[0];
            }
            
            auto pred = model.forward(batch_input);
            auto loss = dl::ops::binary_cross_entropy(pred, batch_target);
            loss->grad().assign(1, 1.0f);
            dl::ComputationGraph::getInstance().backward();
            model.update_parameters(learning_rate);
            
            // Track accuracy
            int batch_correct = 0;
            float batch_loss = loss->data()[0];
            
            for (size_t i = 0; i < current_batch_size; ++i) {
                bool correct_prediction = (pred->data()[i] >= 0.5f) == (batch_target.data()[i] >= 0.5f);
                if (correct_prediction) batch_correct++;
            }
            
            total_loss += batch_loss;
            correct += batch_correct;
        }
        
        float avg_loss = total_loss / inputs.size();
        float accuracy = 100.0f * correct / inputs.size();
        best_accuracy = std::max(best_accuracy, accuracy);
        
        if ((epoch + 1) % 25 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << 50
                     << " - Loss: " << std::fixed << std::setprecision(6) << avg_loss 
                     << " - Accuracy: " << accuracy << "% "
                     << " - Best: " << best_accuracy << "%" << std::endl;
        }
        
        // Early stopping check
        if (avg_loss >= prev_loss) {
            patience++;
            if (patience >= max_patience) {
                std::cout << "\nEarly stopping triggered after " << epoch + 1 << " epochs" << std::endl;
                std::cout << "Best accuracy: " << best_accuracy << "%" << std::endl;
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
