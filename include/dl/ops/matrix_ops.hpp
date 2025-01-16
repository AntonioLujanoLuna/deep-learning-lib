#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include "../utils/utils.hpp"
#include <stdexcept>
#include <memory>
#include <algorithm> // for std::fill
#include <iostream>  // for debug prints (optional)

namespace dl {
namespace ops {

//
// MatMulNode
//
template<typename T>
class MatMulNode : public Node {
public:
    MatMulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output);

    std::string node_type() const override {
        return "MatMul";
    }

    void backward() override;

private:
    // We store the input and output Tensors as shared_ptr
    std::shared_ptr<Tensor<T>> a_;
    std::shared_ptr<Tensor<T>> b_;
    std::shared_ptr<Tensor<T>> output_;
};

// Constructor: sets up DAG links
template<typename T>
MatMulNode<T>::MatMulNode(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& output)
    : a_(std::make_shared<Tensor<T>>(a))
    , b_(std::make_shared<Tensor<T>>(b))
    , output_(std::make_shared<Tensor<T>>(output))
{
    // 1) This node is the gradFn for 'output'
    output.setGradFn(this->shared_from_this());

    // 2) If 'a' was created by another Node, link it as a parent
    if (auto parentA = (*a).gradFn().lock()) {
        parents_.push_back(parentA);
        parentA->children_.push_back(this->shared_from_this());
    }

    // 3) If 'b' was created by another Node, link it as a parent
    if (auto parentB = (*b).gradFn().lock()) {
        parents_.push_back(parentB);
        parentB->children_.push_back(this->shared_from_this());
    }

    // 4) Register this node with the ComputationGraph
    ComputationGraph::getInstance().addNode(this->shared_from_this());
}

template<typename T>
void MatMulNode<T>::backward() {
    std::cout << "\n=== Starting MatMul backward ===" << std::endl;
    std::cout << "A shape: " << utils::shape_to_string(a_->shape()) << std::endl;
    std::cout << "B shape: " << utils::shape_to_string(b_->shape()) << std::endl;
    std::cout << "Output shape: " << utils::shape_to_string(output_->shape()) << std::endl;
    
    const auto& out_grad = output_->grad();
    const auto& a_shape = a_->shape();
    const auto& b_shape = b_->shape();
    
    // Validate output gradient
    if (out_grad.empty() || out_grad.size() != a_shape[0] * b_shape[1]) {
        std::cout << "Invalid output gradient. Size: " << out_grad.size()
                  << ", Expected: " << (a_shape[0] * b_shape[1]) << std::endl;
        throw std::runtime_error("Cannot access gradient: tensor has empty shape");
    }
    
    // Debug: print first few grads
    std::cout << "First few output grads: ";
    for (size_t i = 0; i < std::min<size_t>(3, out_grad.size()); ++i) {
        std::cout << out_grad[i] << " ";
    }
    std::cout << std::endl;
    
    // Clear or zero out existing gradients for A and B
    if (a_->requires_grad()) {
        std::cout << "Clearing A gradients..." << std::endl;
        auto& a_grad = a_->grad();
        std::fill(a_grad.begin(), a_grad.end(), T(0));
    }
    if (b_->requires_grad()) {
        std::cout << "Clearing B gradients..." << std::endl;
        auto& b_grad = b_->grad();
        std::fill(b_grad.begin(), b_grad.end(), T(0));
    }
    
    // ---------------------------
    // Compute gradients for A
    // ---------------------------
    if (a_->requires_grad()) {
        std::cout << "Computing gradients for A" << std::endl;
        auto& a_grad = a_->grad();
        const auto& b_data = b_->data();
        
        // Ensure a_grad is sized properly
        if (a_grad.size() != a_shape[0] * a_shape[1]) {
            std::cout << "Resizing A gradient from " << a_grad.size()
                      << " to " << (a_shape[0] * a_shape[1]) << std::endl;
            a_grad.resize(a_shape[0] * a_shape[1], T(0));
        }
        
        // dL/dA = dL/dC * B^T
        for (size_t i = 0; i < a_shape[0]; ++i) {   // M
            for (size_t j = 0; j < a_shape[1]; ++j) {  // K
                for (size_t k = 0; k < b_shape[1]; ++k) {  // N
                    a_grad[i * a_shape[1] + j] +=
                        out_grad[i * b_shape[1] + k] * b_data[j * b_shape[1] + k];
                }
            }
        }
        
        // Print first few A grads
        std::cout << "First few A grads: ";
        for (size_t i = 0; i < std::min<size_t>(3, a_grad.size()); ++i) {
            std::cout << a_grad[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // ---------------------------
    // Compute gradients for B
    // ---------------------------
    if (b_->requires_grad()) {
        std::cout << "Computing gradients for B" << std::endl;
        auto& b_grad = b_->grad();
        const auto& a_data = a_->data();
        
        // Ensure b_grad is sized properly
        if (b_grad.size() != b_shape[0] * b_shape[1]) {
            std::cout << "Resizing B gradient from " << b_grad.size()
                      << " to " << (b_shape[0] * b_shape[1]) << std::endl;
            b_grad.resize(b_shape[0] * b_shape[1], T(0));
        }
        
        // dL/dB = A^T * dL/dC
        for (size_t i = 0; i < b_shape[0]; ++i) {   // K
            for (size_t j = 0; j < b_shape[1]; ++j) {  // N
                for (size_t k = 0; k < a_shape[0]; ++k) {  // M
                    b_grad[i * b_shape[1] + j] +=
                        a_data[k * a_shape[1] + i] * out_grad[k * b_shape[1] + j];
                }
            }
        }
        
        // Print first few B grads
        std::cout << "First few B grads: ";
        for (size_t i = 0; i < std::min<size_t>(3, b_grad.size()); ++i) {
            std::cout << b_grad[i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "=== MatMul backward completed ===" << std::endl;
}

//---------------------------------------------------------------------------
// matmul(...) free function
//---------------------------------------------------------------------------
template<typename T>
std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("MatMul requires 2D tensors");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error(
            "Invalid shapes for matrix multiplication. A: [" 
            + std::to_string(a.shape()[0]) + ", " 
            + std::to_string(a.shape()[1]) + "], B: ["
            + std::to_string(b.shape()[0]) + ", "
            + std::to_string(b.shape()[1]) + "]"
        );
    }

    // Create output tensor
    auto output = std::make_shared<Tensor<T>>(
        std::vector<size_t>{a.shape()[0], b.shape()[1]}
    );
    output->set_requires_grad(true);
    output->grad().resize(output->data().size(), T(0));

    // Get data references
    const auto& a_data = a.data();
    const auto& b_data = b.data();
    auto& output_data = output->data();

    // Compute matrix multiplication (forward pass)
    const size_t M = a.shape()[0];  // Rows of A
    const size_t K = a.shape()[1];  // Cols of A = Rows of B
    const size_t N = b.shape()[1];  // Cols of B

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            output_data[i * N + j] = sum;
        }
    }

    // If either input requires grad, create a MatMulNode
    if (a.requires_grad() || b.requires_grad()) {
        // The node constructor will do the DAG linking + register itself
        auto node = std::make_shared<MatMulNode<T>>(a, b, *output);
        // We do *not* call addNode again, since the constructor does it
    }

    return output;
}

//---------------------------------------------------------------------------
// Conv2DNode
//---------------------------------------------------------------------------
template<typename T>
class Conv2DNode : public Node {
public:
    Conv2DNode(const Tensor<T>& input, const Tensor<T>& kernel, 
               Tensor<T>& result, size_t stride, size_t padding);

    std::string node_type() const override {
        return "Conv2D";
    }

    void backward() override;

private:
    std::shared_ptr<Tensor<T>> input_;
    std::shared_ptr<Tensor<T>> kernel_;
    std::shared_ptr<Tensor<T>> result_;
    size_t stride_;
    size_t padding_;
};

// Constructor: sets up DAG links
template<typename T>
Conv2DNode<T>::Conv2DNode(const Tensor<T>& input, 
                          const Tensor<T>& kernel, 
                          Tensor<T>& result,
                          size_t stride, 
                          size_t padding)
    : input_(std::make_shared<Tensor<T>>(input))
    , kernel_(std::make_shared<Tensor<T>>(kernel))
    , result_(std::make_shared<Tensor<T>>(result))
    , stride_(stride)
    , padding_(padding)
{
    // 1) This node is gradFn for 'result'
    result.setGradFn(this->shared_from_this());

    // 2) If input has a gradFn, link as parent
    if (auto parentInput = (*input).gradFn().lock()) {
        parents_.push_back(parentInput);
        parentInput->children_.push_back(this->shared_from_this());
    }

    // 3) If kernel has a gradFn, link as parent
    if (auto parentKernel = (*kernel).gradFn().lock()) {
        parents_.push_back(parentKernel);
        parentKernel->children_.push_back(this->shared_from_this());
    }

    // 4) Register node in the ComputationGraph
    ComputationGraph::getInstance().addNode(this->shared_from_this());
}

template<typename T>
void Conv2DNode<T>::backward() {
    const auto& result_grad = result_->grad();
    const auto& input_shape = input_->shape();
    const auto& kernel_shape = kernel_->shape();
    
    // If input_ requires grad, compute partial w.r.t. input
    if (input_->requires_grad()) {
        auto& input_grad = input_->grad();
        const auto& kernel_data = kernel_->data();
        
        // (Optional) You might want to zero out input_grad
        // if you keep it from earlier computations

        // Compute input gradients
        for (size_t h = 0; h < result_->shape()[2]; ++h) {
            for (size_t w = 0; w < result_->shape()[3]; ++w) {
                for (size_t c = 0; c < kernel_shape[1]; ++c) {
                    for (size_t kh = 0; kh < kernel_shape[2]; ++kh) {
                        for (size_t kw = 0; kw < kernel_shape[3]; ++kw) {
                            size_t in_h = h * stride_ + kh - padding_;
                            size_t in_w = w * stride_ + kw - padding_;
                            
                            if (in_h < input_shape[2] && in_w < input_shape[3]) {
                                input_grad[c * input_shape[2] * input_shape[3] +
                                           in_h * input_shape[3] + in_w] 
                                    += kernel_data[c * kernel_shape[2] * kernel_shape[3] +
                                                   kh * kernel_shape[3] + kw] 
                                       * result_grad[h * result_->shape()[3] + w];
                            }
                        }
                    }
                }
            }
        }
    }

    // If kernel_ requires grad, compute partial w.r.t. kernel
    if (kernel_->requires_grad()) {
        auto& kernel_grad = kernel_->grad();
        const auto& input_data = input_->data();
        
        // (Optional) zero out kernel_grad

        // Compute kernel gradients
        for (size_t h = 0; h < result_->shape()[2]; ++h) {
            for (size_t w = 0; w < result_->shape()[3]; ++w) {
                for (size_t c = 0; c < kernel_shape[1]; ++c) {
                    for (size_t kh = 0; kh < kernel_shape[2]; ++kh) {
                        for (size_t kw = 0; kw < kernel_shape[3]; ++kw) {
                            size_t in_h = h * stride_ + kh - padding_;
                            size_t in_w = w * stride_ + kw - padding_;
                            
                            if (in_h < input_shape[2] && in_w < input_shape[3]) {
                                kernel_grad[c * kernel_shape[2] * kernel_shape[3] +
                                            kh * kernel_shape[3] + kw]
                                    += input_data[c * input_shape[2] * input_shape[3] +
                                                  in_h * input_shape[3] + in_w]
                                       * result_grad[h * result_->shape()[3] + w];
                            }
                        }
                    }
                }
            }
        }
    }
}

//---------------------------------------------------------------------------
// conv2d(...) free function
//---------------------------------------------------------------------------
template<typename T>
Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel,
                 size_t stride = 1, size_t padding = 0) {
    const auto& input_shape = input.shape();
    const auto& kernel_shape = kernel.shape();
    
    if (input_shape.size() != 4 || kernel_shape.size() != 4) {
        throw std::runtime_error("Input and kernel must be 4D tensors");
    }
    
    size_t batch_size   = input_shape[0];
    size_t out_channels = kernel_shape[0];
    size_t out_height   = (input_shape[2] + 2 * padding - kernel_shape[2]) / stride + 1;
    size_t out_width    = (input_shape[3] + 2 * padding - kernel_shape[3]) / stride + 1;
    
    // Create output tensor
    auto result = std::make_shared<Tensor<T>>(
        std::vector<size_t>{batch_size, out_channels, out_height, out_width}
    );
    result->set_requires_grad(input.requires_grad() || kernel.requires_grad());

    // Perform the forward pass (convolution)
    auto& result_data  = result->data();
    const auto& input_data  = input.data();
    const auto& kernel_data = kernel.data();

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
                                    sum += input_data[b * input_shape[1]*input_shape[2]*input_shape[3]
                                                    + c * input_shape[2]*input_shape[3]
                                                    + in_h * input_shape[3]
                                                    + in_w]
                                         * kernel_data[oc * kernel_shape[1]*kernel_shape[2]*kernel_shape[3]
                                                     + c * kernel_shape[2]*kernel_shape[3]
                                                     + kh * kernel_shape[3]
                                                     + kw];
                                }
                            }
                        }
                    }
                    result_data[b * out_channels * out_height * out_width
                              + oc * out_height * out_width
                              + h * out_width
                              + w] = sum;
                }
            }
        }
    }
    
    // If input or kernel require grad, create Conv2DNode
    if (input.requires_grad() || kernel.requires_grad()) {
        auto node = std::make_shared<Conv2DNode<T>>(input, kernel, *result, stride, padding);
        // The constructor does the DAG linking & addNode
    }

    // Return by value
    return *result;
}

} // namespace ops
} // namespace dl
