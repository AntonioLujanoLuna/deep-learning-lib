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
        : input_(input), weights_(weights), bias_(bias), output_(output) {}

    void backward() override {
        const auto& output_grad = output_.grad();
        const size_t batch_size = input_.shape()[0];
        const size_t in_features = input_.shape()[1];
        const size_t out_features = weights_.shape()[1];

        std::cout << "Linear backward - Input shape: [" << batch_size << " " << in_features << "]" << std::endl;
        std::cout << "Weights shape: [" << weights_.shape()[0] << " " << weights_.shape()[1] << "]" << std::endl;
        std::cout << "Output grad shape: [" << output_.shape()[0] << " " << output_.shape()[1] << "]" << std::endl;
        
        std::cout << "First few output grads: ";
        for (size_t i = 0; i < std::min(size_t(3), output_grad.size()); ++i) {
            std::cout << output_grad[i] << " ";
        }
        std::cout << std::endl;

        // Compute gradients for weights if needed
        if (weights_.requires_grad()) {
            auto& weights_grad = const_cast<Tensor<T>&>(weights_).grad();
            const auto& input_data = input_.data();

            // dL/dW = X^T * dL/dY
            for (size_t i = 0; i < in_features; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    T sum = T(0);
                    for (size_t b = 0; b < batch_size; ++b) {
                        sum += input_data[b * in_features + i] * output_grad[b * out_features + j];
                    }
                    weights_grad[i * out_features + j] += sum;
                }
            }

            std::cout << "First few weight grads: ";
            for (size_t i = 0; i < std::min(size_t(3), weights_grad.size()); ++i) {
                std::cout << weights_grad[i] << " ";
            }
            std::cout << std::endl;
        }

        // Compute gradients for bias if needed
        if (bias_.requires_grad()) {
            auto& bias_grad = const_cast<Tensor<T>&>(bias_).grad();

            // dL/db = sum(dL/dY, dim=0)
            for (size_t j = 0; j < out_features; ++j) {
                T sum = T(0);
                for (size_t b = 0; b < batch_size; ++b) {
                    sum += output_grad[b * out_features + j];
                }
                bias_grad[j] += sum;
            }

            std::cout << "First few bias grads: ";
            for (size_t i = 0; i < std::min(size_t(3), bias_grad.size()); ++i) {
                std::cout << bias_grad[i] << " ";
            }
            std::cout << std::endl;
        }

        // Compute gradients for input if needed
        if (input_.requires_grad()) {
            auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
            const auto& weights_data = weights_.data();

            // dL/dX = dL/dY * W^T
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < in_features; ++i) {
                    T sum = T(0);
                    for (size_t j = 0; j < out_features; ++j) {
                        sum += output_grad[b * out_features + j] * weights_data[i * out_features + j];
                    }
                    input_grad[b * in_features + i] += sum;
                }
            }

            std::cout << "First few input grads: ";
            for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
                std::cout << input_grad[i] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    const Tensor<T>& input_;
    const Tensor<T>& weights_;
    const Tensor<T>& bias_;
    Tensor<T>& output_;
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
        
        weights_.set_requires_grad(true);
        bias_.set_requires_grad(true);
    }

    Tensor<T> forward(const Tensor<T>& input) {
        // Check input dimensions
        if (input.shape().size() != 2 || input.shape()[1] != in_features_) {
            throw std::runtime_error("Invalid input shape for linear layer");
        }
        
        // y = xW + b
        auto output = ops::matmul(input, weights_);
        output.set_requires_grad(true);
        
        // Add bias to each row
        for (size_t i = 0; i < output.shape()[0]; ++i) {
            for (size_t j = 0; j < output.shape()[1]; ++j) {
                output.data()[i * output.shape()[1] + j] += bias_.data()[j];
            }
        }

        // Create node for backward pass
        auto node = std::make_shared<LinearNode<T>>(input, weights_, bias_, output);
        ComputationGraph::getInstance().addNode(node);
        
        return output;
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
