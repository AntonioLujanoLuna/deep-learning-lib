#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include <cmath>
#include <memory>

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
            const auto& input_data = input_.data();
            const auto& output_grad = output_.grad();
            
            for (size_t i = 0; i < input_data.size(); ++i) {
                input_grad[i] += output_grad[i] * (input_data[i] > 0 ? 1 : 0);
            }
        }
    }

private:
    const Tensor<T>& input_;
    const Tensor<T>& output_;
};

template<typename T>
class SigmoidNode : public Node {
public:
    SigmoidNode(const Tensor<T>& input, Tensor<T>& output)
        : input_(input), output_(output) {}

    void backward() override {
        if (input_.requires_grad()) {
            auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
            const auto& output_data = output_.data();
            const auto& output_grad = output_.grad();
            
            for (size_t i = 0; i < output_data.size(); ++i) {
                input_grad[i] += output_grad[i] * output_data[i] * (1 - output_data[i]);
            }
        }
    }

private:
    const Tensor<T>& input_;
    const Tensor<T>& output_;
};

template<typename T>
class TanhNode : public Node {
public:
    TanhNode(const Tensor<T>& input, Tensor<T>& output)
        : input_(input), output_(output) {}

    void backward() override {
        if (input_.requires_grad()) {
            auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
            const auto& output_data = output_.data();
            const auto& output_grad = output_.grad();
            
            for (size_t i = 0; i < output_data.size(); ++i) {
                input_grad[i] += output_grad[i] * (1 - output_data[i] * output_data[i]);
            }
        }
    }

private:
    const Tensor<T>& input_;
    const Tensor<T>& output_;
};

template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    Tensor<T> output(input.shape());
    auto& output_data = output.data();
    const auto& input_data = input.data();
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
    }
    
    auto node = std::make_shared<ReLUNode<T>>(input, output);
    ComputationGraph::getInstance().addNode(node);
    
    return output;
}

template<typename T>
Tensor<T> sigmoid(const Tensor<T>& input) {
    Tensor<T> output(input.shape());
    auto& output_data = output.data();
    const auto& input_data = input.data();
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = T(1) / (T(1) + std::exp(-input_data[i]));
    }
    
    auto node = std::make_shared<SigmoidNode<T>>(input, output);
    ComputationGraph::getInstance().addNode(node);
    
    return output;
}

template<typename T>
Tensor<T> tanh(const Tensor<T>& input) {
    Tensor<T> output(input.shape());
    auto& output_data = output.data();
    const auto& input_data = input.data();
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::tanh(input_data[i]);
    }
    
    auto node = std::make_shared<TanhNode<T>>(input, output);
    ComputationGraph::getInstance().addNode(node);
    
    return output;
}

} // namespace ops
} // namespace dl
