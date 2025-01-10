#pragma once

#include "../tensor.hpp"
#include "../ops/matrix_ops.hpp"
#include "../autograd.hpp"
#include "../optim/optimizer.hpp"
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
        : input_tensor_(std::make_shared<Tensor<T>>(input))
        , weights_tensor_(std::make_shared<Tensor<T>>(weights))
        , bias_tensor_(std::make_shared<Tensor<T>>(bias))
        , output_tensor_(std::make_shared<Tensor<T>>(output)) {}

    std::string node_type() const override {
        return "Linear";
    }

    void backward() override {
        const auto& output_grad = output_tensor_->grad();
        const size_t batch_size = input_tensor_->shape()[0];
        const size_t in_features = input_tensor_->shape()[1];
        const size_t out_features = weights_tensor_->shape()[1];

        // Compute input gradients if needed
        if (input_tensor_->requires_grad()) {
            auto& input_grad = input_tensor_->grad();
            const auto& weights_data = weights_tensor_->data();
            
            // Initialize gradient
            input_grad.assign(batch_size * in_features, T(0));
            
            // dL/dx = dL/dy * W^T
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < in_features; ++j) {
                    for (size_t k = 0; k < out_features; ++k) {
                        input_grad[i * in_features + j] += output_grad[i * out_features + k] * weights_data[j * out_features + k];
                    }
                }
            }
        }

        // Compute weight gradients if needed
        if (weights_tensor_->requires_grad()) {
            auto& weights_grad = weights_tensor_->grad();
            const auto& input_data = input_tensor_->data();
            
            // Initialize gradient
            weights_grad.assign(in_features * out_features, T(0));
            
            // dL/dW = x^T * dL/dy
            for (size_t i = 0; i < in_features; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    for (size_t k = 0; k < batch_size; ++k) {
                        weights_grad[i * out_features + j] += input_data[k * in_features + i] * output_grad[k * out_features + j];
                    }
                }
            }
        }

        // Compute bias gradients if needed
        if (bias_tensor_->requires_grad()) {
            auto& bias_grad = bias_tensor_->grad();
            
            // Initialize gradient
            bias_grad.assign(out_features, T(0));
            
            // dL/db = sum(dL/dy, dim=0)
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    bias_grad[j] += output_grad[i * out_features + j];
                }
            }
        }
    }

private:
    std::shared_ptr<Tensor<T>> input_tensor_;
    std::shared_ptr<Tensor<T>> weights_tensor_;
    std::shared_ptr<Tensor<T>> bias_tensor_;
    std::shared_ptr<Tensor<T>> output_tensor_;
};

template<typename T>
class Linear {
public:
    Linear(size_t in_features, size_t out_features) 
        : in_features_(in_features)
        , out_features_(out_features)
        , weights_({in_features, out_features})
        , bias_({1, out_features}) {
        
        // Xavier/Glorot initialization
        T scale = std::sqrt(T(2) / (in_features + out_features));
        init_weights(scale);
        
        weights_.set_requires_grad(true);
        bias_.set_requires_grad(true);
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
        if (input.shape()[1] != weights_.shape()[0]) {  
            throw std::runtime_error("Input feature dimension (" + std::to_string(input.shape()[1]) + 
                                   ") must match weight input dimension (" + std::to_string(weights_.shape()[0]) + ")");
        }
        
        // Perform matrix multiplication
        auto output = dl::ops::matmul(input, weights_);
        
        // Add bias
        for (size_t i = 0; i < output->data().size(); ++i) {
            output->data()[i] += bias_.data()[i % bias_.data().size()];
        }
        
        // Set requires_grad if needed
        if (input.requires_grad() || weights_.requires_grad() || bias_.requires_grad()) {
            output->set_requires_grad(true);
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
