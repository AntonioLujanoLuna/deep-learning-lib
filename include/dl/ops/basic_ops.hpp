#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"

namespace dl {
namespace ops {

template<typename T>
class AddNode : public Node {
public:
    AddNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
        : a_tensor_(a), b_tensor_(b), output_tensor_(output) {}

    std::string node_type() const override {
        return "Add";
    }

    void backward() override {
        if (a_tensor_.requires_grad()) {
            auto& a_grad = a_tensor_.grad();
            const auto& out_grad = output_tensor_.grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                a_grad[i] += out_grad[i];
            }
        }
        
        if (b_tensor_.requires_grad()) {
            auto& b_grad = b_tensor_.grad();
            const auto& out_grad = output_tensor_.grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                b_grad[i] += out_grad[i];
            }
        }
    }

private:
    Tensor<T> a_tensor_;  // Store by value to keep tensors alive
    Tensor<T> b_tensor_;
    Tensor<T> output_tensor_;
};

template<typename T>
class MulNode : public Node {
public:
    MulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
        : a_tensor_(a), b_tensor_(b), output_tensor_(output) {}

    std::string node_type() const override {
        return "Multiply";
    }

    void backward() override {
        const auto& out_grad = output_tensor_.grad();
        
        if (a_tensor_.requires_grad()) {
            auto& a_grad = a_tensor_.grad();
            const auto& b_data = b_tensor_.data();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                a_grad[i] += out_grad[i] * b_data[i];
            }
        }
        
        if (b_tensor_.requires_grad()) {
            auto& b_grad = b_tensor_.grad();
            const auto& a_data = a_tensor_.data();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                b_grad[i] += out_grad[i] * a_data[i];
            }
        }
    }

private:
    Tensor<T> a_tensor_;  // Store by value to keep tensors alive
    Tensor<T> b_tensor_;
    Tensor<T> output_tensor_;
};

} // namespace ops
} // namespace dl
