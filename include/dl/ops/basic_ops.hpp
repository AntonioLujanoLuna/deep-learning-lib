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

// Template implementations
template<typename T>
AddNode<T>::AddNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
    : a_impl_(a.impl_)
    , b_impl_(b.impl_)
    , output_impl_(output.impl_) {}

template<typename T>
void AddNode<T>::backward() {
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

template<typename T>
MulNode<T>::MulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
    : a_impl_(a.impl_)
    , b_impl_(b.impl_)
    , output_impl_(output.impl_) {}

template<typename T>
void MulNode<T>::backward() {
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
