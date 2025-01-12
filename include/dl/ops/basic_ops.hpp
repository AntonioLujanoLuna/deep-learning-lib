#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include "../detail/tensor_impl.hpp"
#include <memory>
#include <vector>

namespace dl {
namespace ops {

template<typename T>
class AddNode : public Node {
public:
    AddNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
        : a_impl_(a.impl_)
        , b_impl_(b.impl_)
        , output_impl_(output.impl_) {}

    std::string node_type() const override {
        return "Add";
    }

    void backward() override {
        const auto& out_grad = output_impl_->grad();
        
        if (a_impl_->requires_grad()) {
            auto& a_grad = a_impl_->grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                a_grad[i] += out_grad[i];  // Add gradient contribution
            }
        }
        
        if (b_impl_->requires_grad()) {
            auto& b_grad = b_impl_->grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                b_grad[i] += out_grad[i];  // Add gradient contribution
            }
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> a_impl_;
    std::shared_ptr<detail::TensorImpl<T>> b_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
};

template<typename T>
class MulNode : public Node {
public:
    MulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
        : a_impl_(a.impl_)
        , b_impl_(b.impl_)
        , output_impl_(output.impl_) {}

    std::string node_type() const override {
        return "Multiply";
    }

    void backward() override {
        const auto& out_grad = output_impl_->grad();
        const auto& a_data = a_impl_->data();
        const auto& b_data = b_impl_->data();
        
        if (a_impl_->requires_grad()) {
            auto& a_grad = a_impl_->grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                a_grad[i] += out_grad[i] * b_data[i];  // Chain rule: dy/da = b * upstream_grad
            }
        }
        
        if (b_impl_->requires_grad()) {
            auto& b_grad = b_impl_->grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                b_grad[i] += out_grad[i] * a_data[i];  // Chain rule: dy/db = a * upstream_grad
            }
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> a_impl_;
    std::shared_ptr<detail::TensorImpl<T>> b_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
};

} // namespace ops
} // namespace dl
