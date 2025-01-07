#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include <stdexcept>
#include <memory>

namespace dl {
namespace ops {

template<typename T>
class MatMulNode : public Node {
public:
    MatMulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
        : a_(a), b_(b), output_(output) {}

    void backward() override {
        const auto& output_grad = output_.grad();
        const size_t m = a_.shape()[0];
        const size_t k = a_.shape()[1];
        const size_t n = b_.shape()[1];

        // Compute gradient for a if needed
        if (a_.requires_grad()) {
            auto& a_grad = const_cast<Tensor<T>&>(a_).grad();
            const auto& b_data = b_.data();

            // dL/dA = dL/dY * B^T
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    T sum = T(0);
                    for (size_t l = 0; l < n; ++l) {
                        sum += output_grad[i * n + l] * b_data[j * n + l];
                    }
                    a_grad[i * k + j] += sum;
                }
            }
        }

        // Compute gradient for b if needed
        if (b_.requires_grad()) {
            auto& b_grad = const_cast<Tensor<T>&>(b_).grad();
            const auto& a_data = a_.data();

            // dL/dB = A^T * dL/dY
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    T sum = T(0);
                    for (size_t l = 0; l < m; ++l) {
                        sum += a_data[l * k + i] * output_grad[l * n + j];
                    }
                    b_grad[i * n + j] += sum;
                }
            }
        }
    }

private:
    const Tensor<T>& a_;
    const Tensor<T>& b_;
    Tensor<T>& output_;
};

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    // Check dimensions
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Invalid shapes for matrix multiplication");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];
    const size_t n = b.shape()[1];

    Tensor<T> output({m, n});
    auto& output_data = output.data();
    const auto& a_data = a.data();
    const auto& b_data = b.data();

    // Compute matrix multiplication
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t l = 0; l < k; ++l) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            output_data[i * n + j] = sum;
        }
    }

    if (a.requires_grad() || b.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<MatMulNode<T>>(a, b, output);
        ComputationGraph::getInstance().addNode(node);
    }

    return output;
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
