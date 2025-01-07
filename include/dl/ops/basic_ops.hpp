#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"

namespace dl {
namespace ops {

template<typename T>
class AddNode : public Node {
public:
    AddNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
        : a_(a), b_(b), output_(output) {}

    std::string node_type() const override {
        return "Add";
    }

    void backward() override {
        if (a_.requires_grad()) {
            auto& a_grad = const_cast<Tensor<T>&>(a_).grad();
            const auto& out_grad = output_.grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                a_grad[i] += out_grad[i];
            }
        }
        
        if (b_.requires_grad()) {
            auto& b_grad = const_cast<Tensor<T>&>(b_).grad();
            const auto& out_grad = output_.grad();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                b_grad[i] += out_grad[i];
            }
        }
    }

private:
    const Tensor<T>& a_;  // Store by reference to keep tensors alive
    const Tensor<T>& b_;
    Tensor<T>& output_;
};

template<typename T>
class MulNode : public Node {
public:
    MulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
        : a_(a), b_(b), output_(output) {}

    std::string node_type() const override {
        return "Multiply";
    }

    void backward() override {
        const auto& out_grad = output_.grad();
        
        if (a_.requires_grad()) {
            auto& a_grad = const_cast<Tensor<T>&>(a_).grad();
            const auto& b_data = b_.data();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                a_grad[i] += out_grad[i] * b_data[i];
            }
        }
        
        if (b_.requires_grad()) {
            auto& b_grad = const_cast<Tensor<T>&>(b_).grad();
            const auto& a_data = a_.data();
            for (size_t i = 0; i < out_grad.size(); ++i) {
                b_grad[i] += out_grad[i] * a_data[i];
            }
        }
    }

private:
    const Tensor<T>& a_;  // Store by reference to keep tensors alive
    const Tensor<T>& b_;
    Tensor<T>& output_;
};

} // namespace ops
} // namespace dl
