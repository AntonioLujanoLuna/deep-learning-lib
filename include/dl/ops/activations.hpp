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
    ReLUNode(const Tensor<T>& input, std::shared_ptr<Tensor<T>> output)
        : input_(input), output_(output) {
        std::cout << "\n=== Creating ReLU node ===" << std::endl;
        std::cout << "Input shape: " << utils::shape_to_string(input_.shape()) << std::endl;
        std::cout << "Output shape: " << utils::shape_to_string(output_->shape()) << std::endl;
    }

    std::string node_type() const override {
        return "ReLU";
    }

    void backward() override {
        std::cout << "\n=== Starting ReLU backward ===" << std::endl;
        if (!input_.requires_grad()) {
            std::cout << "Input does not require gradients, skipping backward" << std::endl;
            return;
        }

        auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
        const auto& output_grad = output_->grad();
        const auto& input_data = input_.data();

        std::cout << "Input shape: " << utils::shape_to_string(input_.shape()) << std::endl;
        std::cout << "Output grad shape: " << utils::shape_to_string(output_->shape()) << std::endl;

        // Validate gradient sizes
        if (input_grad.size() != input_data.size()) {
            std::cout << "Resizing input gradient from " << input_grad.size() 
                      << " to " << input_data.size() << std::endl;
            input_grad.resize(input_data.size(), T(0));
        }

        // dL/dx = dL/dy if x > 0, 0 otherwise
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_grad[i] += (input_data[i] > T(0)) ? output_grad[i] : T(0);
        }

        std::cout << "First few input grads: ";
        for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
            std::cout << input_grad[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "=== ReLU backward completed ===" << std::endl;
    }

private:
    const Tensor<T>& input_;
    std::shared_ptr<Tensor<T>> output_;
};

template<typename T>
class SigmoidNode : public Node {
public:
    SigmoidNode(const Tensor<T>& input, Tensor<T>& output)
        : input_(input), output_(output) {
        
        std::cout << "\nSigmoidNode constructor:" << std::endl;
        std::cout << "  Input tensor:" << std::endl;
        std::cout << "    Shape: " << utils::shape_to_string(input_.shape()) << std::endl;
        std::cout << "    Requires grad: " << std::boolalpha << input_.requires_grad() << std::endl;
        std::cout << "    Data size: " << input_.data().size() << std::endl;
        if (input_.requires_grad()) {
            std::cout << "    Grad size: " << input_.grad().size() << std::endl;
        }
        
        std::cout << "  Output tensor:" << std::endl;
        std::cout << "    Shape: " << utils::shape_to_string(output_.shape()) << std::endl;
        std::cout << "    Data size: " << output_.data().size() << std::endl;
    }

    std::string node_type() const override {
        return "Sigmoid";
    }

    void backward() override {
        std::cout << "\n=== Starting Sigmoid backward ===" << std::endl;
        if (input_.requires_grad()) {
            std::cout << "Input tensor requires gradients" << std::endl;
            
            std::cout << "Input tensor state:" << std::endl;
            std::cout << "  Shape: " << utils::shape_to_string(input_.shape()) << std::endl;
            std::cout << "  Data size: " << input_.data().size() << std::endl;
            
            auto& input_grad = const_cast<Tensor<T>&>(input_).grad();
            std::cout << "  Grad size: " << input_grad.size() << std::endl;
            
            const auto& output_grad = output_.grad();
            const auto& output_data = output_.data();
            
            std::cout << "Output tensor state:" << std::endl;
            std::cout << "  Shape: " << utils::shape_to_string(output_.shape()) << std::endl;
            std::cout << "  Data size: " << output_data.size() << std::endl;
            std::cout << "  Grad size: " << output_grad.size() << std::endl;
            
            // Validate sizes
            if (input_grad.size() != output_grad.size() || output_data.size() != output_grad.size()) {
                throw std::runtime_error("Size mismatch in Sigmoid backward pass: input_grad=" + 
                    std::to_string(input_grad.size()) + ", output_grad=" + 
                    std::to_string(output_grad.size()) + ", output_data=" + 
                    std::to_string(output_data.size()));
            }
            
            std::cout << "Computing gradients..." << std::endl;
            for (size_t i = 0; i < output_data.size(); ++i) {
                // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                input_grad[i] += output_grad[i] * output_data[i] * (T(1) - output_data[i]);
                
                if (i < 3) {
                    std::cout << "i=" << i << ": output_data=" << output_data[i] 
                              << ", output_grad=" << output_grad[i]
                              << ", input_grad=" << input_grad[i] << std::endl;
                }
            }
            
            std::cout << "First few input grads after Sigmoid:" << std::endl;
            for (size_t i = 0; i < std::min(size_t(3), input_grad.size()); ++i) {
                std::cout << "grad[" << i << "] = " << input_grad[i] << std::endl;
            }
        } else {
            std::cout << "Input tensor does not require gradients" << std::endl;
        }
        std::cout << "=== Sigmoid backward completed ===" << std::endl;
    }

private:
    const Tensor<T>& input_;
    Tensor<T>& output_;
};

template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    std::cout << "\n=== Starting ReLU forward ===" << std::endl;
    std::cout << "Input requires_grad: " << std::boolalpha << input.requires_grad() << std::endl;

    // Create output tensor and store in graph
    auto& graph = ComputationGraph::getInstance();
    std::shared_ptr<Tensor<T>> output = std::make_shared<Tensor<T>>(input.shape());
    output->set_requires_grad(input.requires_grad());

    // Store tensors in computation graph
    graph.storeTensor(input);
    graph.storeTensorPtr(output);

    // Compute ReLU
    const auto& input_data = input.data();
    auto& output_data = output->data();
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::max(input_data[i], T(0));
    }

    // Create node for backward pass if needed
    if (input.requires_grad()) {
        std::cout << "Setting output requires_grad to true (input requires gradients)" << std::endl;
        auto node = std::make_shared<ReLUNode<T>>(input, output);
        graph.addNode(node);
    } else {
        std::cout << "Output does not require gradients (input does not require gradients)" << std::endl;
    }

    std::cout << "Output shape: " << utils::shape_to_string(output->shape()) << std::endl;
    std::cout << "Output requires_grad: " << output->requires_grad() << std::endl;
    std::cout << "=== ReLU forward completed ===" << std::endl;
    return *output;
}

template<typename T>
Tensor<T> sigmoid(const Tensor<T>& input) {
    const auto& input_data = input.data();
    auto& graph = ComputationGraph::getInstance();
    
    // Store input tensor in graph to keep it alive
    graph.storeTensor(input);
    
    // Create output tensor
    std::shared_ptr<Tensor<T>> output = std::make_shared<Tensor<T>>(input.shape());
    output->set_requires_grad(input.requires_grad());
    auto& output_data = output->data();
    
    // Store output tensor in graph to keep it alive
    graph.storeTensorPtr(output);
    
    // Compute sigmoid
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = T(1) / (T(1) + std::exp(-input_data[i]));
    }
    
    if (input.requires_grad()) {
        auto node = std::make_shared<SigmoidNode<T>>(input, *output);
        graph.addNode(node);
    }
    
    return *output;
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
