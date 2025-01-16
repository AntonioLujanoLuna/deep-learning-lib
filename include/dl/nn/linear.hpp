#pragma once

#include "dl/autograd.hpp"
#include "dl/tensor.hpp"
#include <random>
#include <memory>
#include <iostream>
#include <vector>

namespace dl {
namespace nn {

template<typename T>
class LinearNode : public Node {
public:
    LinearNode(const Tensor<T>& input, const Tensor<T>& weights, const Tensor<T>& bias, Tensor<T>& output)
        : input_impl_(input.impl_)
        , weights_impl_(weights.impl_)
        , bias_impl_(bias.impl_)
        , output_impl_(output.impl_) {}

    std::string node_type() const override {
        return "Linear";
    }

    void backward() override {
        const auto& output_grad = output_impl_->grad();
        const size_t batch_size = input_impl_->shape()[0];
        const size_t in_features = input_impl_->shape()[1];
        const size_t out_features = weights_impl_->shape()[0];  

        // Compute input gradients if needed
        if (input_impl_->requires_grad()) {
            auto& input_grad = input_impl_->grad();
            const auto& weights_data = weights_impl_->data();
            
            // Initialize gradient
            input_grad.assign(batch_size * in_features, T(0));
            
            // dL/dx = dL/dy * W^T
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < in_features; ++j) {
                    for (size_t k = 0; k < out_features; ++k) {
                        input_grad[i * in_features + j] += output_grad[i * out_features + k] * weights_data[k * in_features + j];  
                    }
                }
            }
        }

        // Compute weight gradients if needed
        if (weights_impl_->requires_grad()) {
            auto& weights_grad = weights_impl_->grad();
            const auto& input_data = input_impl_->data();
            
            // Initialize gradient
            weights_grad.assign(out_features * in_features, T(0));
            
            // dL/dW = x^T * dL/dy
            for (size_t i = 0; i < out_features; ++i) {
                for (size_t j = 0; j < in_features; ++j) {
                    for (size_t k = 0; k < batch_size; ++k) {
                        weights_grad[i * in_features + j] += input_data[k * in_features + j] * output_grad[k * out_features + i];
                    }
                }
            }
        }

        // Compute bias gradients if needed
        if (bias_impl_->requires_grad()) {
            auto& bias_grad = bias_impl_->grad();
            
            // Initialize gradient
            bias_grad.assign(out_features, T(0));
            
            // dL/db = sum(dL/dy) over batch dimension
            for (size_t i = 0; i < out_features; ++i) {
                for (size_t j = 0; j < batch_size; ++j) {
                    bias_grad[i] += output_grad[j * out_features + i];
                }
            }
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> weights_impl_;
    std::shared_ptr<detail::TensorImpl<T>> bias_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
};

template<typename T>
class Linear {
public:
    Linear(size_t in_features, size_t out_features) 
        : in_features_(in_features)
        , out_features_(out_features)
        , weights_({out_features, in_features})  
        , bias_({1, out_features}) {
        
        // Xavier/Glorot initialization
        T scale = std::sqrt(T(2) / (in_features + out_features));
        init_weights(scale);
        
        weights_.set_requires_grad(true);
        bias_.set_requires_grad(true);
        
        // Initialize gradients
        weights_.grad().assign(weights_.data().size(), T(0));
        bias_.grad().assign(bias_.data().size(), T(0));
    }

    void init_weights(T scale) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0, scale);
        
        auto& w_data = weights_.data();
        for (size_t i = 0; i < w_data.size(); ++i) {
            w_data[i] = dist(gen);
        }
        
        auto& b_data = bias_.data();
        for (size_t i = 0; i < b_data.size(); ++i) {
            b_data[i] = 0;  // Initialize bias to zero
        }
    }

    std::shared_ptr<Tensor<T>> forward(const Tensor<T>& input) {
        // Check input dimensions
        if (input.shape().size() != 2) {
            throw std::runtime_error("Input tensor must be 2D (batch_size x input_features)");
        }
        if (input.shape()[1] != weights_.shape()[1]) {  
            throw std::runtime_error("Input feature dimension (" + std::to_string(input.shape()[1]) + 
                                   ") must match weight input dimension (" + std::to_string(weights_.shape()[1]) + ")");
        }
        
        // Create output tensor
        auto output = std::make_shared<Tensor<T>>(std::vector<size_t>{input.shape()[0], out_features_});
        
        // Perform matrix multiplication manually since we don't have transpose
        const auto& input_data = input.data();
        const auto& weights_data = weights_.data();
        auto& output_data = output->data();
        
        const size_t batch_size = input.shape()[0];
        const size_t in_features = input.shape()[1];
        
        // Initialize output to zero
        output_data.assign(batch_size * out_features_, T(0));
        
        // Compute output = input * weights^T
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features_; ++j) {
                for (size_t k = 0; k < in_features; ++k) {
                    output_data[i * out_features_ + j] += input_data[i * in_features + k] * weights_data[j * in_features + k];
                }
            }
        }
        
        // Add bias (broadcasting across batch dimension)
        const auto& bias_data = bias_.data();
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features_; ++j) {
                output_data[i * out_features_ + j] += bias_data[j];
            }
        }
        
        // Create backward node
        if (input.requires_grad() || weights_.requires_grad() || bias_.requires_grad()) {
            output->set_requires_grad(true);
            auto node = std::make_shared<LinearNode<T>>(input, weights_, bias_, *output);
            ComputationGraph::getInstance().addNode(node);
        }
        
        return output;
    }

    void zero_grad() {
        weights_.grad().assign(weights_.grad().size(), T(0));
        bias_.grad().assign(bias_.grad().size(), T(0));
    }

    const Tensor<T>& weights() const { return weights_; }
    Tensor<T>& weights() { return weights_; }
    const Tensor<T>& bias() const { return bias_; }
    Tensor<T>& bias() { return bias_; }

    // Get all parameters for optimization
    std::vector<std::reference_wrapper<Tensor<T>>> parameters() {
        return {weights_, bias_};
    }

    // Add all parameters to optimizer
    void add_parameters_to_optimizer(optim::SGD<T>& optimizer) {
        optimizer.add_parameter(weights_);
        optimizer.add_parameter(bias_);
    }

private:
    size_t in_features_;
    size_t out_features_;
    Tensor<T> weights_;
    Tensor<T> bias_;
};

} // namespace nn
} // namespace dl
