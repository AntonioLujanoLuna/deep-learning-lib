#pragma once

#include "dl/tensor.hpp"
#include "dl/nn/linear.hpp"
#include "dl/ops/activations.hpp"
#include <vector>
#include <memory>

namespace dl {
namespace nn {

template<typename T>
class MLP {
public:
    MLP(const std::vector<size_t>& layer_sizes) {
        if (layer_sizes.size() < 2) {
            throw std::runtime_error("MLP must have at least input and output layers");
        }

        // Create layers
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            layers.push_back(std::make_shared<Linear<T>>(layer_sizes[i], layer_sizes[i + 1]));
            layers.back()->weights().set_requires_grad(true);
            layers.back()->bias().set_requires_grad(true);
        }
    }

    std::shared_ptr<Tensor<T>> forward(const Tensor<T>& x) {
        std::shared_ptr<Tensor<T>> current = std::make_shared<Tensor<T>>(x);
        
        for (size_t i = 0; i < layers.size(); ++i) {
            current = layers[i]->forward(*current);
            
            // Apply ReLU to all layers except the last one
            if (i < layers.size() - 1) {
                current = dl::ops::relu(*current);
            } else {
                // Apply sigmoid to the last layer for binary classification
                current = dl::ops::sigmoid(*current);
            }
        }
        
        return current;
    }

    void zero_grad() {
        for (auto& layer : layers) {
            layer->zero_grad();
        }
    }

    void update_parameters(T learning_rate) {
        for (auto& layer : layers) {
            auto& weights = layer->weights();
            auto& bias = layer->bias();
            const auto& w_grad = weights.grad();
            const auto& b_grad = bias.grad();

            // Update weights
            for (size_t i = 0; i < weights.data().size(); ++i) {
                weights.data()[i] -= learning_rate * w_grad[i];
            }

            // Update bias
            for (size_t i = 0; i < bias.data().size(); ++i) {
                bias.data()[i] -= learning_rate * b_grad[i];
            }
        }
    }

    // Get total number of parameters
    size_t num_parameters() const {
        size_t total = 0;
        for (const auto& layer : layers) {
            total += layer->weights().data().size() + layer->bias().data().size();
        }
        return total;
    }

    // Get layer at specific index
    std::shared_ptr<Linear<T>> get_layer(size_t index) const {
        if (index >= layers.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers[index];
    }

private:
    std::vector<std::shared_ptr<Linear<T>>> layers;
};

} // namespace nn
} // namespace dl
