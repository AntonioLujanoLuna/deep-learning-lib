#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"

namespace dl {
namespace ops {

template<typename T>
class AddNode : public Node {
public:
    AddNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& result)
        : a_(a), b_(b), result_(result) {}

    void backward() override {
        if (a_.requires_grad()) {
            auto& a_grad = const_cast<Tensor<T>&>(a_).grad();
            const auto& result_grad = result_.grad();
            for (size_t i = 0; i < a_grad.size(); ++i) {
                a_grad[i] += result_grad[i];
            }
        }
        
        if (b_.requires_grad()) {
            auto& b_grad = const_cast<Tensor<T>&>(b_).grad();
            const auto& result_grad = result_.grad();
            for (size_t i = 0; i < b_grad.size(); ++i) {
                b_grad[i] += result_grad[i];
            }
        }
    }

private:
    Tensor<T> a_;  // Store by value to keep tensors alive
    Tensor<T> b_;
    Tensor<T> result_;
};

template<typename T>
class MulNode : public Node {
public:
    MulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& result)
        : a_(a), b_(b), result_(result) {}

    void backward() override {
        if (a_.requires_grad()) {
            auto& a_grad = const_cast<Tensor<T>&>(a_).grad();
            const auto& result_grad = result_.grad();
            const auto& b_data = b_.data();
            for (size_t i = 0; i < a_grad.size(); ++i) {
                a_grad[i] += result_grad[i] * b_data[i];
            }
        }
        
        if (b_.requires_grad()) {
            auto& b_grad = const_cast<Tensor<T>&>(b_).grad();
            const auto& result_grad = result_.grad();
            const auto& a_data = a_.data();
            for (size_t i = 0; i < b_grad.size(); ++i) {
                b_grad[i] += result_grad[i] * a_data[i];
            }
        }
    }

private:
    Tensor<T> a_;  // Store by value to keep tensors alive
    Tensor<T> b_;
    Tensor<T> result_;
};

} // namespace ops
} // namespace dl
