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
    LinearNode(Tensor<T>& input, Tensor<T>& weights, Tensor<T>& bias, Tensor<T>& output)
        : input_tensor_(input), weights_tensor_(weights), bias_tensor_(bias), output_tensor_(output) {}

    void backward() override {
        std::cout << "\n=== Starting Linear backward ===" << std::endl;
        const auto& output_grad = output_tensor_.grad();
        const size_t batch_size = input_tensor_.shape()[0];
        const size_t in_features = input_tensor_.shape()[1];
        const size_t out_features = weights_tensor_.shape()[1];

        std::cout << "Linear backward - Input shape: [" << batch_size << ", " << in_features << "]" << std::endl;
        std::cout << "Weights shape: [" << weights_tensor_.shape()[0] << ", " << weights_tensor_.shape()[1] << "]" << std::endl;
        std::cout << "Output grad shape: [" << output_tensor_.shape()[0] << ", " << output_tensor_.shape()[1] << "]" << std::endl;
        std::cout << "Input requires_grad: " << input_tensor_.requires_grad() << std::endl;
        std::cout << "Weights requires_grad: " << weights_tensor_.requires_grad() << std::endl;
        std::cout << "Bias requires_grad: " << bias_tensor_.requires_grad() << std::endl;
        
        // Validate output gradient
        if (output_grad.empty() || output_grad.size() != batch_size * out_features) {
            std::cout << "Invalid output gradient. Size: " << output_grad.size() 
                      << ", Expected: " << (batch_size * out_features) << std::endl;
            return;
        }
        
        std::cout << "First few output grads: ";
        for (size_t i = 0; i < std::min(size_t(3), output_grad.size()); ++i) {
            std::cout << output_grad[i] << " ";
        }
        std::cout << std::endl;

        // Compute input gradients if needed
        if (input_tensor_.requires_grad()) {
            std::cout << "Computing input gradients..." << std::endl;
            auto& input_grad = input_tensor_.grad();
            const auto& weights_data = weights_tensor_.data();
            
            // Initialize input gradients if needed
            if (input_grad.size() != batch_size * in_features) {
                std::cout << "Resizing input gradient from " << input_grad.size() 
                          << " to " << (batch_size * in_features) << std::endl;
                input_grad.resize(batch_size * in_features, T(0));
            }
            
            // Clear input gradients first since we're computing them from scratch
            std::fill(input_grad.begin(), input_grad.end(), T(0));
            // dL/dX = dL/dY * W^T
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < in_features; ++i) {
                    for (size_t j = 0; j < out_features; ++j) {
                        input_grad[b * in_features + i] += weights_data[i * out_features + j] * output_grad[b * out_features + j];
                    }
                }
            }
            
            std::cout << "First few input grads: ";
            for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
                std::cout << input_grad[i] << " ";
            }
            std::cout << std::endl;
        }

        // Compute gradients for weights if needed
        if (weights_tensor_.requires_grad()) {
            std::cout << "Computing weight gradients..." << std::endl;
            auto& weights_grad = weights_tensor_.grad();
            const auto& input_data = input_tensor_.data();
            
            // Validate gradient size
            if (weights_grad.size() != in_features * out_features) {
                std::cout << "Resizing weights gradient from " << weights_grad.size() 
                          << " to " << (in_features * out_features) << std::endl;
                weights_grad.resize(in_features * out_features, T(0));
            }

            // Clear weight gradients first since we're computing them from scratch
            std::fill(weights_grad.begin(), weights_grad.end(), T(0));
            // dL/dW = X^T * dL/dY
            for (size_t i = 0; i < in_features; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    for (size_t b = 0; b < batch_size; ++b) {
                        weights_grad[i * out_features + j] += input_data[b * in_features + i] * output_grad[b * out_features + j];
                    }
                }
            }

            std::cout << "First few weight grads: ";
            for (size_t i = 0; i < std::min(size_t(3), weights_grad.size()); ++i) {
                std::cout << weights_grad[i] << " ";
            }
            std::cout << std::endl;
        }

        // Compute gradients for bias if needed
        if (bias_tensor_.requires_grad()) {
            std::cout << "Computing bias gradients..." << std::endl;
            auto& bias_grad = bias_tensor_.grad();
            
            // Validate gradient size
            if (bias_grad.size() != out_features) {
                std::cout << "Resizing bias gradient from " << bias_grad.size() 
                          << " to " << out_features << std::endl;
                bias_grad.resize(out_features, T(0));
            }

            // Clear bias gradients first since we're computing them from scratch
            std::fill(bias_grad.begin(), bias_grad.end(), T(0));
            // dL/db = sum(dL/dY, dim=0)
            for (size_t j = 0; j < out_features; ++j) {
                for (size_t b = 0; b < batch_size; ++b) {
                    bias_grad[j] += output_grad[b * out_features + j];
                }
            }

            std::cout << "First few bias grads: ";
            for (size_t i = 0; i < std::min(size_t(3), bias_grad.size()); ++i) {
                std::cout << bias_grad[i] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "=== Linear backward completed ===" << std::endl;
    }

    std::string node_type() const override {
        return "Linear";
    }

private:
    Tensor<T>& input_tensor_;
    Tensor<T>& weights_tensor_;
    Tensor<T>& bias_tensor_;
    Tensor<T>& output_tensor_;
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

    Tensor<T> forward(const Tensor<T>& input) {
        //std::cout << "\n=== Starting Linear forward ===" << std::endl;
        //std::cout << "Input requires_grad: " << std::boolalpha << input.requires_grad() << std::endl;
        //std::cout << "Weights requires_grad: " << weights_.requires_grad() << std::endl;
        //std::cout << "Bias requires_grad: " << bias_.requires_grad() << std::endl;
        
        // Check input dimensions
        if (input.shape().size() != 2 || input.shape()[1] != in_features_) {
            throw std::runtime_error("Invalid input shape for linear layer. Expected [N, " + 
                std::to_string(in_features_) + "], got " + utils::shape_to_string(input.shape()));
        }
        
        // Initialize weights and bias gradients
        weights_.grad().resize(weights_.data().size(), T(0));
        bias_.grad().resize(bias_.data().size(), T(0));
        
        // Create output tensor with proper shape
        std::vector<size_t> output_shape = {input.shape()[0], out_features_};
        std::shared_ptr<Tensor<T>> output = std::make_shared<Tensor<T>>(output_shape);
        output->set_requires_grad(true);
        output->grad().resize(input.shape()[0] * out_features_, T(0));  // Initialize gradient vector
        
        // Store tensors in computation graph
        auto& graph = ComputationGraph::getInstance();
        graph.storeTensor(input);
        graph.storeTensor(weights_);
        graph.storeTensor(bias_);
        graph.storeTensorPtr(output);
        
        // y = xW + b
        *output = ops::matmul(input, weights_);
        
        // Add bias to each row
        const auto& bias_data = bias_.data();
        auto& output_data = output->data();
        const size_t batch_size = input.shape()[0];
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features_; ++j) {
                output_data[i * out_features_ + j] += bias_data[j];
            }
        }

        // Create node for backward pass
        auto stored_input = std::make_shared<Tensor<T>>(input);  // Store a copy of the input
        graph.storeTensorPtr(stored_input);
        auto node = std::make_shared<LinearNode<T>>(*stored_input, weights_, bias_, *output);
        graph.addNode(node);
        
        //std::cout << "Output shape: " << utils::shape_to_string(output->shape()) << std::endl;
        //std::cout << "Output requires_grad: " << output->requires_grad() << std::endl;
        //std::cout << "=== Linear forward completed ===" << std::endl;
        return *output;
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
