#pragma once

#include "../autograd.hpp"
#include <memory>
#include <vector>
#include <string>

namespace dl {

// Forward declarations
template<typename T> class Tensor;

namespace detail {
template<typename T> class TensorImpl;
}

namespace ops {

template<typename T>
class AddNode : public Node {
public:
    AddNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output);

    std::string node_type() const override { return "Add"; }
    void backward() override;

private:
    std::shared_ptr<detail::TensorImpl<T>> a_impl_;
    std::shared_ptr<detail::TensorImpl<T>> b_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
};

template<typename T>
class MulNode : public Node {
public:
    MulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output);

    std::string node_type() const override { return "Multiply"; }
    void backward() override;

private:
    std::shared_ptr<detail::TensorImpl<T>> a_impl_;
    std::shared_ptr<detail::TensorImpl<T>> b_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
};

// ------------------------------------------------------
// Implementation for AddNode
// ------------------------------------------------------
template<typename T>
AddNode<T>::AddNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
    : a_impl_(a.impl_)
    , b_impl_(b.impl_)
    , output_impl_(output.impl_)
{
    // 1) This node is the gradFn for 'output'
    output.setGradFn(this->shared_from_this());

    // 2) If 'a' was created by some Node, link as parent
    if (auto parentA = a.gradFn().lock()) {
        parents_.push_back(parentA);
        parentA->children_.push_back(this->shared_from_this());
    }
    // 3) If 'b' was created by some Node, link as parent
    if (auto parentB = b.gradFn().lock()) {
        parents_.push_back(parentB);
        parentB->children_.push_back(this->shared_from_this());
    }

    // 4) Register this node with the ComputationGraph
    ComputationGraph::getInstance().addNode(this->shared_from_this());
}

template<typename T>
void AddNode<T>::backward() {
    // The gradient of (a + b) w.r.t. a is 1
    // The gradient of (a + b) w.r.t. b is 1
    const auto& out_grad = output_impl_->grad();
    
    if (a_impl_->requires_grad()) {
        auto& a_grad = a_impl_->grad();
        for (size_t i = 0; i < out_grad.size(); ++i) {
            a_grad[i] += out_grad[i];
        }
    }
    
    if (b_impl_->requires_grad()) {
        auto& b_grad = b_impl_->grad();
        for (size_t i = 0; i < out_grad.size(); ++i) {
            b_grad[i] += out_grad[i];
        }
    }
}

// ------------------------------------------------------
// Implementation for MulNode
// ------------------------------------------------------
template<typename T>
MulNode<T>::MulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
    : a_impl_(a.impl_)
    , b_impl_(b.impl_)
    , output_impl_(output.impl_)
{
    // 1) This node is the gradFn for 'output'
    output.setGradFn(this->shared_from_this());

    // 2) If 'a' was created by some Node, link as parent
    if (auto parentA = a.gradFn().lock()) {
        parents_.push_back(parentA);
        parentA->children_.push_back(this->shared_from_this());
    }
    // 3) If 'b' was created by some Node, link as parent
    if (auto parentB = b.gradFn().lock()) {
        parents_.push_back(parentB);
        parentB->children_.push_back(this->shared_from_this());
    }

    // 4) Register this node with the ComputationGraph
    ComputationGraph::getInstance().addNode(this->shared_from_this());
}

template<typename T>
void MulNode<T>::backward() {
    // The gradient of (a * b) w.r.t. a is b
    // The gradient of (a * b) w.r.t. b is a
    const auto& out_grad = output_impl_->grad();
    const auto& a_data = a_impl_->data();
    const auto& b_data = b_impl_->data();
    
    if (a_impl_->requires_grad()) {
        auto& a_grad = a_impl_->grad();
        for (size_t i = 0; i < out_grad.size(); ++i) {
            a_grad[i] += out_grad[i] * b_data[i];
        }
    }
    
    if (b_impl_->requires_grad()) {
        auto& b_grad = b_impl_->grad();
        for (size_t i = 0; i < out_grad.size(); ++i) {
            b_grad[i] += out_grad[i] * a_data[i];
        }
    }
}

} // namespace ops
} // namespace dl
