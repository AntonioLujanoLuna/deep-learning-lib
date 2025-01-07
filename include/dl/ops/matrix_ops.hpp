#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include <stdexcept>

namespace dl {
namespace ops {

template<typename T>
class MatMulNode : public Node {
public:
    MatMulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& result)
        : a_(a), b_(b), result_(result) {
        if (a.shape().size() != 2 || b.shape().size() != 2) {
            throw std::runtime_error("MatMul requires 2D tensors");
        }
        if (a.shape()[1] != b.shape()[0]) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }
    }

    void backward() override {
        const size_t M = a_.shape()[0];
        const size_t K = a_.shape()[1];
        const size_t N = b_.shape()[1];

        if (a_.requires_grad()) {
            auto& a_grad = const_cast<Tensor<T>&>(a_).grad();
            const auto& result_grad = result_.grad();
            const auto& b_data = b_.data();

            // dC/dA = dC/dY * B^T
            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    T sum = T(0);
                    for (size_t j = 0; j < N; ++j) {
                        sum += result_grad[i * N + j] * b_data[k * N + j];
                    }
                    a_grad[i * K + k] += sum;
                }
            }
        }

        if (b_.requires_grad()) {
            auto& b_grad = const_cast<Tensor<T>&>(b_).grad();
            const auto& result_grad = result_.grad();
            const auto& a_data = a_.data();

            // dC/dB = A^T * dC/dY
            for (size_t k = 0; k < K; ++k) {
                for (size_t j = 0; j < N; ++j) {
                    T sum = T(0);
                    for (size_t i = 0; i < M; ++i) {
                        sum += a_data[i * K + k] * result_grad[i * N + j];
                    }
                    b_grad[k * N + j] += sum;
                }
            }
        }
    }

private:
    Tensor<T> a_;  // Store by value to keep tensors alive
    Tensor<T> b_;
    Tensor<T> result_;
};

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("MatMul requires 2D tensors");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }

    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];
    const size_t N = b.shape()[1];

    Tensor<T> result({M, N});
    auto& result_data = result.data();
    const auto& a_data = a.data();
    const auto& b_data = b.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }

    // Set requires_grad if either input requires grad
    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<MatMulNode<T>>(a, b, result);
        ComputationGraph::getInstance().addNode(node);
    }

    return result;
}

template<typename T>
class Conv2DNode : public Node {
public:
    Conv2DNode(const Tensor<T>& input, const Tensor<T>& kernel, 
               Tensor<T>& result, size_t stride, size_t padding)
        : input_(input), kernel_(kernel), result_(result)
        , stride_(stride), padding_(padding) {}

    void backward() override {
        const auto& result_grad = result_.grad();
        const auto& input_shape = input_.shape();
        const auto& kernel_shape = kernel_.shape();
        
        if (input_.requires_grad()) {
            auto& input_grad = input_.grad();
            const auto& kernel_data = kernel_.data();
            
            // Zero padding for input gradients
            std::vector<T> padded_grad(input_grad.size(), T(0));
            
            // Compute input gradients
            for (size_t h = 0; h < result_.shape()[2]; ++h) {
                for (size_t w = 0; w < result_.shape()[3]; ++w) {
                    for (size_t c = 0; c < kernel_shape[1]; ++c) {
                        for (size_t kh = 0; kh < kernel_shape[2]; ++kh) {
                            for (size_t kw = 0; kw < kernel_shape[3]; ++kw) {
                                size_t in_h = h * stride_ + kh - padding_;
                                size_t in_w = w * stride_ + kw - padding_;
                                
                                if (in_h < input_shape[2] && in_w < input_shape[3]) {
                                    input_grad[c * input_shape[2] * input_shape[3] + 
                                             in_h * input_shape[3] + in_w] +=
                                        kernel_data[c * kernel_shape[2] * kernel_shape[3] +
                                                  kh * kernel_shape[3] + kw] *
                                        result_grad[h * result_.shape()[3] + w];
                                }
                            }
                        }
                    }
                }
            }
        }

        if (kernel_.requires_grad()) {
            auto& kernel_grad = kernel_.grad();
            const auto& input_data = input_.data();
            
            // Compute kernel gradients
            for (size_t h = 0; h < result_.shape()[2]; ++h) {
                for (size_t w = 0; w < result_.shape()[3]; ++w) {
                    for (size_t c = 0; c < kernel_shape[1]; ++c) {
                        for (size_t kh = 0; kh < kernel_shape[2]; ++kh) {
                            for (size_t kw = 0; kw < kernel_shape[3]; ++kw) {
                                size_t in_h = h * stride_ + kh - padding_;
                                size_t in_w = w * stride_ + kw - padding_;
                                
                                if (in_h < input_shape[2] && in_w < input_shape[3]) {
                                    kernel_grad[c * kernel_shape[2] * kernel_shape[3] +
                                              kh * kernel_shape[3] + kw] +=
                                        input_data[c * input_shape[2] * input_shape[3] +
                                                 in_h * input_shape[3] + in_w] *
                                        result_grad[h * result_.shape()[3] + w];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

private:
    Tensor<T> input_;  // Store by value to keep tensors alive
    Tensor<T> kernel_;
    Tensor<T> result_;
    size_t stride_;
    size_t padding_;
};

template<typename T>
Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel,
                 size_t stride = 1, size_t padding = 0) {
    const auto& input_shape = input.shape();
    const auto& kernel_shape = kernel.shape();
    
    if (input_shape.size() != 4 || kernel_shape.size() != 4) {
        throw std::runtime_error("Input and kernel must be 4D tensors");
    }
    
    size_t batch_size = input_shape[0];
    size_t out_channels = kernel_shape[0];
    size_t out_height = (input_shape[2] + 2 * padding - kernel_shape[2]) / stride + 1;
    size_t out_width = (input_shape[3] + 2 * padding - kernel_shape[3]) / stride + 1;
    
    Tensor<T> result({batch_size, out_channels, out_height, out_width});
    auto& result_data = result.data();
    const auto& input_data = input.data();
    const auto& kernel_data = kernel.data();
    
    // Perform convolution
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t h = 0; h < out_height; ++h) {
                for (size_t w = 0; w < out_width; ++w) {
                    T sum = 0;
                    for (size_t c = 0; c < kernel_shape[1]; ++c) {
                        for (size_t kh = 0; kh < kernel_shape[2]; ++kh) {
                            for (size_t kw = 0; kw < kernel_shape[3]; ++kw) {
                                size_t in_h = h * stride + kh - padding;
                                size_t in_w = w * stride + kw - padding;
                                
                                if (in_h < input_shape[2] && in_w < input_shape[3]) {
                                    sum += input_data[b * input_shape[1] * input_shape[2] * input_shape[3] +
                                                    c * input_shape[2] * input_shape[3] +
                                                    in_h * input_shape[3] + in_w] *
                                         kernel_data[oc * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] +
                                                   c * kernel_shape[2] * kernel_shape[3] +
                                                   kh * kernel_shape[3] + kw];
                                }
                            }
                        }
                    }
                    result_data[b * out_channels * out_height * out_width +
                              oc * out_height * out_width +
                              h * out_width + w] = sum;
                }
            }
        }
    }
    
    if (input.requires_grad() || kernel.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<Conv2DNode<T>>(input, kernel, result, stride, padding);
        ComputationGraph::getInstance().addNode(node);
    }

    return result;
}

} // namespace ops
} // namespace dl
