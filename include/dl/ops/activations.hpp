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
    ReLUNode(Tensor<T>& input, std::shared_ptr<Tensor<T>> output)
        : input_tensor_(input), output_tensor_(output) {
        //std::cout << "\n=== Creating ReLU node ===" << std::endl;
        //std::cout << "Input shape: " << utils::shape_to_string(input_tensor_.shape()) << std::endl;
        //std::cout << "Output shape: " << utils::shape_to_string(output_tensor_->shape()) << std::endl;
    }

    std::string node_type() const override {
        return "ReLU";
    }

    void backward() override {
        std::cout << "\n=== Starting ReLU backward ===" << std::endl;
        if (!input_tensor_.requires_grad()) {
            std::cout << "Input does not require gradients, skipping backward" << std::endl;
            return;
        }

        auto& input_grad = input_tensor_.grad();
        const auto& output_grad = output_tensor_->grad();
        const auto& input_data = input_tensor_.data();

        std::cout << "Input shape: " << utils::shape_to_string(input_tensor_.shape()) << std::endl;
        std::cout << "Output grad shape: " << utils::shape_to_string(output_tensor_->shape()) << std::endl;
        std::cout << "Input grad size: " << input_grad.size() << std::endl;
        std::cout << "Output grad size: " << output_grad.size() << std::endl;

        // Initialize gradient if needed
        if (input_grad.size() != input_data.size()) {
            input_grad.resize(input_data.size(), T(0));
        }

        // dL/dx = dL/dy if x > 0, 0 otherwise
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_grad[i] = (input_data[i] > T(0)) ? output_grad[i] : T(0);  // Use = instead of +=
        }

        std::cout << "First few input grads after ReLU backward: ";
        for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
            std::cout << input_grad[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "=== ReLU backward completed ===" << std::endl;
    }

private:
    Tensor<T>& input_tensor_;
    std::shared_ptr<Tensor<T>> output_tensor_;
};

template<typename T>
class SigmoidNode : public Node {
public:
    SigmoidNode(Tensor<T>& input, Tensor<T>& output)
        : input_tensor_(input), output_tensor_(output) {
        
        //std::cout << "\nSigmoidNode constructor:" << std::endl;
        //std::cout << "  Input tensor:" << std::endl;
        //std::cout << "    Shape: " << utils::shape_to_string(input_tensor_.shape()) << std::endl;
        //std::cout << "    Requires grad: " << std::boolalpha << input_tensor_.requires_grad() << std::endl;
        //std::cout << "    Data size: " << input_tensor_.data().size() << std::endl;
        if (input_tensor_.requires_grad()) {
            //std::cout << "    Grad size: " << input_tensor_.grad().size() << std::endl;
        }
        
        //std::cout << "  Output tensor:" << std::endl;
        //std::cout << "    Shape: " << utils::shape_to_string(output_tensor_.shape()) << std::endl;
        //std::cout << "    Data size: " << output_tensor_.data().size() << std::endl;
    }

    std::string node_type() const override {
        return "Sigmoid";
    }

    void backward() override {
        std::cout << "\n=== Starting Sigmoid backward ===" << std::endl;
        if (input_tensor_.requires_grad()) {
            std::cout << "Input tensor requires gradients" << std::endl;
            
            std::cout << "Input tensor state:" << std::endl;
            std::cout << "  Shape: " << utils::shape_to_string(input_tensor_.shape()) << std::endl;
            std::cout << "  Data size: " << input_tensor_.data().size() << std::endl;
            
            auto& input_grad = input_tensor_.grad();
            std::cout << "  Grad size: " << input_grad.size() << std::endl;
            
            const auto& output_grad = output_tensor_.grad();
            const auto& output_data = output_tensor_.data();

            // Initialize gradient if needed
            if (input_grad.size() != input_tensor_.data().size()) {
                input_grad.resize(input_tensor_.data().size(), T(0));
            }

            // dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))
            for (size_t i = 0; i < output_data.size(); ++i) {
                T sigmoid_x = output_data[i];  // This is already sigmoid(x)
                input_grad[i] = output_grad[i] * sigmoid_x * (T(1) - sigmoid_x);  // Use = instead of += since we're setting the gradient
            }
            
            std::cout << "First few input grads after Sigmoid backward: ";
            for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
                std::cout << input_grad[i] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=== Sigmoid backward completed ===" << std::endl;
    }

private:
    Tensor<T>& input_tensor_;
    Tensor<T>& output_tensor_;
};

template<typename T>
class TanhNode : public Node {
public:
    TanhNode(Tensor<T>& input, Tensor<T>& output)
        : input_tensor_(input), output_tensor_(output) {}

    void backward() override {
        std::cout << "\n=== Starting Tanh backward ===" << std::endl;
        if (input_tensor_.requires_grad()) {
            auto& input_grad = input_tensor_.grad();
            const auto& output_grad = output_tensor_.grad();
            const auto& output_data = output_tensor_.data();
            
            // Initialize gradient if needed
            if (input_grad.size() != input_tensor_.data().size()) {
                input_grad.resize(input_tensor_.data().size(), T(0));
            }
            
            // dL/dx = dL/dy * (1 - y^2)
            for (size_t i = 0; i < output_data.size(); ++i) {
                input_grad[i] = output_grad[i] * (T(1) - output_data[i] * output_data[i]);
            }
            
            std::cout << "First few input grads after Tanh backward: ";
            for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
                std::cout << input_grad[i] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=== Tanh backward completed ===" << std::endl;
    }

private:
    Tensor<T>& input_tensor_;
    Tensor<T>& output_tensor_;
};

template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    //std::cout << "\n=== Starting ReLU forward ===" << std::endl;
    //std::cout << "Input requires_grad: " << std::boolalpha << input.requires_grad() << std::endl;

    // Create output tensor and node
    auto output = std::make_shared<Tensor<T>>(input.shape());
    output->set_requires_grad(true);
    output->grad().resize(input.data().size(), T(0));  // Initialize gradient vector
    
    // Compute ReLU forward pass
    const auto& input_data = input.data();
    auto& output_data = output->data();
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::max(input_data[i], T(0));
    }

    // Create node for backward pass if needed
    if (input.requires_grad()) {
        //std::cout << "Setting output requires_grad to true (input requires gradients)" << std::endl;
        auto node = std::make_shared<ReLUNode<T>>(const_cast<Tensor<T>&>(input), output);
        ComputationGraph::getInstance().addNode(node);
    } else {
        //std::cout << "Output does not require gradients (input does not require gradients)" << std::endl;
    }

    //std::cout << "Output shape: " << utils::shape_to_string(output->shape()) << std::endl;
    //std::cout << "Output requires_grad: " << output->requires_grad() << std::endl;
    //std::cout << "=== ReLU forward completed ===" << std::endl;
    return *output;
}

template<typename T>
Tensor<T> sigmoid(const Tensor<T>& input) {
    const auto& input_data = input.data();
    auto& graph = ComputationGraph::getInstance();
    
    // Store input tensor in graph to keep it alive
    graph.storeTensor(input);
    
    // Create output tensor
    auto output = std::make_shared<Tensor<T>>(input.shape());
    output->set_requires_grad(true);
    output->grad().resize(input_data.size(), T(0));  // Initialize gradient vector
    auto& output_data = output->data();
    
    // Store output tensor in graph
    graph.storeTensorPtr(output);
    
    // Compute sigmoid
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = T(1) / (T(1) + std::exp(-input_data[i]));
    }
    
    if (input.requires_grad()) {
        auto node = std::make_shared<SigmoidNode<T>>(const_cast<Tensor<T>&>(input), *output);
        graph.addNode(node);
    }
    
    return *output;
}

template<typename T>
Tensor<T> tanh(const Tensor<T>& input) {
    const auto& input_data = input.data();
    auto& graph = ComputationGraph::getInstance();
    
    // Store input tensor in graph to keep it alive
    graph.storeTensor(input);
    
    // Create output tensor
    auto output = std::make_shared<Tensor<T>>(input.shape());
    output->set_requires_grad(true);
    output->grad().resize(input_data.size(), T(0));  // Initialize gradient vector
    auto& output_data = output->data();
    
    // Store output tensor in graph
    graph.storeTensorPtr(output);
    
    // Compute tanh
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::tanh(input_data[i]);
    }
    
    if (input.requires_grad()) {
        auto node = std::make_shared<TanhNode<T>>(const_cast<Tensor<T>&>(input), *output);
        graph.addNode(node);
    }
    
    return *output;
}

} // namespace ops
} // namespace dl
