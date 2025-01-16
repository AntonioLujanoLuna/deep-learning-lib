#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include <cmath>
#include <memory>
#include <sstream>

namespace dl {
namespace ops {

//
// ReLUNode
//
template<typename T>
class ReLUNode 
    : public Node
    , public std::enable_shared_from_this<ReLUNode<T>> {
public:
    ReLUNode(const Tensor<T>& input, std::shared_ptr<Tensor<T>> output);

    std::string node_type() const override {
        return "ReLU";
    }

    void backward() override;

private:
    std::shared_ptr<Tensor<T>> input_tensor_;
    std::shared_ptr<Tensor<T>> output_tensor_;
};

//-----------------------------------
// ReLUNode Implementation
//-----------------------------------
template<typename T>
ReLUNode<T>::ReLUNode(const Tensor<T>& input, std::shared_ptr<Tensor<T>> output)
    : input_tensor_(std::make_shared<Tensor<T>>(input))
    , output_tensor_(output)
{
    // 1) The output is produced by *this* ReLUNode
    output->setGradFn(this->shared_from_this());

    // 2) If the input has a gradFn, link it as a parent
    if (auto parent = (*input).gradFn().lock()) {
        parents_.push_back(parent);
        parent->children_.push_back(this->shared_from_this());
    }

    // 3) Register this node with the ComputationGraph
    ComputationGraph::getInstance().addNode(this->shared_from_this());
}

template<typename T>
void ReLUNode<T>::backward() {
    // If input doesn't require grad, there's no need to proceed
    if (!input_tensor_->requires_grad()) {
        std::cout << "Input does not require gradients, skipping backward" << std::endl;
        return;
    }

    // Retrieve references
    auto& input_grad = input_tensor_->grad();
    const auto& output_grad = output_tensor_->grad();
    const auto& input_data = input_tensor_->data();

    // Ensure input_grad is allocated
    if (input_grad.size() != input_data.size()) {
        input_grad.resize(input_data.size(), T(0));
    }

    // ReLU derivative: grad = output_grad if input > 0, else 0
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_grad[i] = (input_data[i] > T(0)) ? output_grad[i] : T(0);
    }
}

//
// SigmoidNode
//
template<typename T>
class SigmoidNode 
    : public Node
    , public std::enable_shared_from_this<SigmoidNode<T>> {
public:
    SigmoidNode(const Tensor<T>& input, std::shared_ptr<Tensor<T>> output);

    std::string node_type() const override {
        return "Sigmoid";
    }

    void backward() override;

private:
    std::shared_ptr<Tensor<T>> input_tensor_;
    std::shared_ptr<Tensor<T>> output_tensor_;
};

//-----------------------------------
// SigmoidNode Implementation
//-----------------------------------
template<typename T>
SigmoidNode<T>::SigmoidNode(const Tensor<T>& input, std::shared_ptr<Tensor<T>> output)
    : input_tensor_(std::make_shared<Tensor<T>>(input))
    , output_tensor_(output)
{
    // 1) The output is produced by *this* SigmoidNode
    output->setGradFn(this->shared_from_this());

    // 2) If the input has a gradFn, link it as a parent
    if (auto parent = (*input).gradFn().lock()) {
        parents_.push_back(parent);
        parent->children_.push_back(this->shared_from_this());
    }

    // 3) Register this node with the ComputationGraph
    ComputationGraph::getInstance().addNode(this->shared_from_this());
}

template<typename T>
void SigmoidNode<T>::backward() {
    if (!input_tensor_->requires_grad()) {
        return;  // no need to do anything
    }
            
    auto& input_grad = input_tensor_->grad();
    const auto& output_grad = output_tensor_->grad();
    const auto& output_data = output_tensor_->data();

    // Ensure input_grad is allocated
    if (input_grad.size() != input_tensor_->data().size()) {
        input_grad.resize(input_tensor_->data().size(), T(0));
    }

    // dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))
    for (size_t i = 0; i < output_data.size(); ++i) {
        T sigmoid_x = output_data[i];  
        input_grad[i] = output_grad[i] * sigmoid_x * (T(1) - sigmoid_x);
    }
}

//
// TanhNode
//
template<typename T>
class TanhNode 
    : public Node
    , public std::enable_shared_from_this<TanhNode<T>> {
public:
    TanhNode(const Tensor<T>& input, std::shared_ptr<Tensor<T>> output);

    std::string node_type() const override {
        return "Tanh";
    }

    void backward() override;

private:
    std::shared_ptr<Tensor<T>> input_tensor_;
    std::shared_ptr<Tensor<T>> output_tensor_;
};

//-----------------------------------
// TanhNode Implementation
//-----------------------------------
template<typename T>
TanhNode<T>::TanhNode(const Tensor<T>& input, std::shared_ptr<Tensor<T>> output)
    : input_tensor_(std::make_shared<Tensor<T>>(input))
    , output_tensor_(output)
{
    // 1) The output is produced by *this* TanhNode
    output->setGradFn(this->shared_from_this());

    // 2) If the input has a gradFn, link it as a parent
    if (auto parent = (*input).gradFn().lock()) {
        parents_.push_back(parent);
        parent->children_.push_back(this->shared_from_this());
    }

    // 3) Register this node with the ComputationGraph
    ComputationGraph::getInstance().addNode(this->shared_from_this());
}

template<typename T>
void TanhNode<T>::backward() {
    std::cout << "\n=== Starting Tanh backward ===" << std::endl;
    if (!input_tensor_->requires_grad()) {
        std::cout << "Input does not require grad, skipping Tanh backward" << std::endl;
        return;
    }

    auto& input_grad = input_tensor_->grad();
    const auto& output_grad = output_tensor_->grad();
    const auto& output_data = output_tensor_->data();

    // Ensure input_grad is allocated
    if (input_grad.size() != input_tensor_->data().size()) {
        input_grad.resize(input_tensor_->data().size(), T(0));
    }
    
    // dL/dx = dL/dy * (1 - y^2), where y = tanh(x)
    for (size_t i = 0; i < output_data.size(); ++i) {
        input_grad[i] = output_grad[i] * (T(1) - output_data[i] * output_data[i]);
    }
    
    // (Optional) Debug prints
    std::cout << "First few input grads after Tanh backward: ";
    for (size_t i = 0; i < std::min<size_t>(3, input_grad.size()); ++i) {
        std::cout << input_grad[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "=== Tanh backward completed ===" << std::endl;
}

//---------------------------------------------------------------------------
// Activation Free Functions: relu, sigmoid, tanh
//---------------------------------------------------------------------------

template<typename T>
std::shared_ptr<Tensor<T>> relu(const Tensor<T>& input) {
    auto output = std::make_shared<Tensor<T>>(input.shape());
    output->set_requires_grad(true);
    output->grad().resize(input.data().size(), T(0));  // Initialize gradient vector

    // Forward pass: ReLU
    const auto& input_data = input.data();
    auto& output_data = output->data();
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::max(input_data[i], T(0));
    }

    // If input requires grad, create a ReLUNode
    if (input.requires_grad()) {
        auto node = std::make_shared<ReLUNode<T>>(input, output);
        // Node constructor does the DAG linking + addNode(...)
    }
    return output;
}

template<typename T>
std::shared_ptr<Tensor<T>> sigmoid(const Tensor<T>& input) {
    auto& graph = ComputationGraph::getInstance();

    // create output
    auto output = std::make_shared<Tensor<T>>(input.shape());
    output->set_requires_grad(true);
    output->grad().resize(input.data().size(), T(0));

    // forward pass
    const auto& input_data = input.data();
    auto& output_data = output->data();
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = T(1) / (T(1) + std::exp(-input_data[i]));
    }

    // If input requires grad, create a SigmoidNode
    if (input.requires_grad()) {
        auto node = std::make_shared<SigmoidNode<T>>(input, output);
        // Node constructor does the DAG linking + addNode(...)
    }

    return output;
}

template<typename T>
Tensor<T> tanh(const Tensor<T>& input) {
    // For consistency, let's also create a shared_ptr output
    auto outputPtr = std::make_shared<Tensor<T>>(input.shape());
    outputPtr->set_requires_grad(true);
    outputPtr->grad().resize(input.data().size(), T(0));

    // forward pass: std::tanh
    const auto& input_data = input.data();
    auto& output_data = outputPtr->data();
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = std::tanh(input_data[i]);
    }

    // If input requires grad, create TanhNode
    if (input.requires_grad()) {
        auto node = std::make_shared<TanhNode<T>>(input, outputPtr);
        // Node constructor does the DAG linking + addNode(...)
    }

    // The function returns by value, so we dereference outputPtr
    return *outputPtr;
}

} // namespace ops
} // namespace dl
