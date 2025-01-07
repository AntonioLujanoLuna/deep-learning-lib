#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include <cmath>
#include <memory>
#include <sstream>

namespace dl {
namespace ops {

template<typename T>
class ReLUNode : public Node {
public:
    ReLUNode(const Tensor<T>& input, Tensor<T>& output)
        : input_(input), output_(output) {}

    void backward() override {
        if (input_.requires_grad()) {
            auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
            const auto& output_grad = output_.grad();
            const auto& input_data = input_.data();
            
            std::cout << "ReLU backward - Input shape: [";
            for (const auto& dim : input_.shape()) {
                std::cout << dim << " ";
            }
            std::cout << "]" << std::endl;
            
            std::cout << "First few output grads: ";
            for (size_t i = 0; i < std::min(size_t(3), output_grad.size()); ++i) {
                std::cout << output_grad[i] << " ";
            }
            std::cout << std::endl;
            
            for (size_t i = 0; i < input_data.size(); ++i) {
                input_grad[i] += output_grad[i] * (input_data[i] > T(0) ? T(1) : T(0));
            }
            
            std::cout << "First few input grads after ReLU: ";
            for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
                std::cout << input_grad[i] << " ";
            }
            std::cout << std::endl << std::flush;
        }
    }

private:
    const Tensor<T>& input_;
    Tensor<T>& output_;
};

template<typename T>
class SigmoidNode : public Node {
public:
    SigmoidNode(const Tensor<T>& input, Tensor<T>& output)
        : input_(input), output_(output) {}

    void backward() override {
        if (input_.requires_grad()) {
            auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
            const auto& output_grad = output_.grad();
            const auto& output_data = output_.data();
            
            std::cout << "Sigmoid backward - Input shape: [";
            for (const auto& dim : input_.shape()) {
                std::cout << dim << " ";
            }
            std::cout << "]" << std::endl;
            
            std::cout << "First few output grads: ";
            for (size_t i = 0; i < std::min(size_t(3), output_grad.size()); ++i) {
                std::cout << output_grad[i] << " ";
            }
            std::cout << std::endl;
            
            for (size_t i = 0; i < output_data.size(); ++i) {
                // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                input_grad[i] += output_grad[i] * output_data[i] * (T(1) - output_data[i]);
            }
            
            std::cout << "First few input grads after Sigmoid: ";
            for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
                std::cout << input_grad[i] << " ";
            }
            std::cout << std::endl << std::flush;
        }
    }

private:
    const Tensor<T>& input_;
    Tensor<T>& output_;
};

template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    const auto& input_data = input.data();
    Tensor<T> output(input.shape());
    output.set_requires_grad(input.requires_grad());
    auto& output_data = output.data();
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::max(input_data[i], T(0));
    }
    
    if (input.requires_grad()) {
        auto node = std::make_shared<ReLUNode<T>>(input, output);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return output;
}

template<typename T>
Tensor<T> sigmoid(const Tensor<T>& input) {
    const auto& input_data = input.data();
    Tensor<T> output(input.shape());
    output.set_requires_grad(input.requires_grad());
    auto& output_data = output.data();
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = T(1) / (T(1) + std::exp(-input_data[i]));
    }
    
    if (input.requires_grad()) {
        auto node = std::make_shared<SigmoidNode<T>>(input, output);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return output;
}

template<typename T>
class TanhNode : public Node {
public:
    TanhNode(const Tensor<T>& input, Tensor<T>& output)
        : input_(input), output_(output) {}

    void backward() override {
        if (input_.requires_grad()) {
            auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
            const auto& output_grad = output_.grad();
            const auto& output_data = output_.data();
            
            for (size_t i = 0; i < output_data.size(); ++i) {
                input_grad[i] += output_grad[i] * (T(1) - output_data[i] * output_data[i]);
            }
        }
    }

private:
    const Tensor<T>& input_;
    Tensor<T>& output_;
};

template<typename T>
Tensor<T> tanh(const Tensor<T>& input) {
    const auto& input_data = input.data();
    Tensor<T> output(input.shape());
    output.set_requires_grad(input.requires_grad());
    auto& output_data = output.data();
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::tanh(input_data[i]);
    }
    
    if (input.requires_grad()) {
        auto node = std::make_shared<TanhNode<T>>(input, output);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return output;
}

} // namespace ops
} // namespace dl
