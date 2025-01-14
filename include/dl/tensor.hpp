#pragma once

#include "detail/tensor_impl.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include <iostream>

// Forward declarations
namespace dl {
namespace ops {
template<typename T> class AddNode;
template<typename T> class MulNode;
template<typename T> class MatMulNode;
template<typename T> class ReLUNode;
template<typename T> class SigmoidNode;
template<typename T> class TanhNode;
}

namespace nn {
template<typename T> class LinearNode;
}

template<typename T>
class Tensor {
public:
    using TensorPtr = std::shared_ptr<detail::TensorImpl<T>>;

    Tensor(const std::vector<size_t>& shape)
        : impl_(std::make_shared<detail::TensorImpl<T>>(shape)) {}

    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
        : impl_(std::make_shared<detail::TensorImpl<T>>(shape, data)) {}

    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    const std::vector<size_t>& shape() const { return impl_->shape(); }
    std::vector<T>& data() { return impl_->data(); }
    const std::vector<T>& data() const { return impl_->data(); }
    std::vector<T>& grad() { return impl_->grad(); }
    const std::vector<T>& grad() const { return impl_->grad(); }

    bool requires_grad() const { return impl_->requires_grad(); }
    void set_requires_grad(bool requires_grad) { impl_->set_requires_grad(requires_grad); }
    void zero_grad() { impl_->zero_grad(); }

    // Forward declarations of operators
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;

private:
    TensorPtr impl_;

    // Give access to operation nodes
    template<typename U> friend class ops::AddNode;
    template<typename U> friend class ops::MulNode;
    template<typename U> friend class ops::MatMulNode;
    template<typename U> friend class ops::ReLUNode;
    template<typename U> friend class ops::SigmoidNode;
    template<typename U> friend class ops::TanhNode;
    template<typename U> friend class nn::LinearNode;
};

} // namespace dl

// Include operator implementations after Tensor class definition
#include "ops/basic_ops.hpp"
#include "ops/broadcast.hpp"
#include "dl/autograd.hpp"

namespace dl {

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    if (!impl_ || !other.impl_) {
        throw std::runtime_error("Tensor implementation is null");
    }

    auto broadcast_shape = ops::compute_broadcast_shape(shape(), other.shape());
    
    // Broadcast inputs if necessary
    std::vector<T> a_data = (shape() == broadcast_shape) ? 
        data() : ops::broadcast_to(data(), shape(), broadcast_shape);
    std::vector<T> b_data = (other.shape() == broadcast_shape) ? 
        other.data() : ops::broadcast_to(other.data(), other.shape(), broadcast_shape);
    
    Tensor<T> result(broadcast_shape);
    auto& result_data = result.data();
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
    
    // Set requires_grad and create backward node if either input requires grad
    if (requires_grad() || other.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<ops::AddNode<T>>(*this, other, result);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
    if (!impl_ || !other.impl_) {
        throw std::runtime_error("Tensor implementation is null");
    }

    auto broadcast_shape = ops::compute_broadcast_shape(shape(), other.shape());
    
    // Broadcast inputs if necessary
    std::vector<T> a_data = (shape() == broadcast_shape) ? 
        data() : ops::broadcast_to(data(), shape(), broadcast_shape);
    std::vector<T> b_data = (other.shape() == broadcast_shape) ? 
        other.data() : ops::broadcast_to(other.data(), other.shape(), broadcast_shape);
    
    Tensor<T> result(broadcast_shape);
    auto& result_data = result.data();
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
    
    // Set requires_grad and create backward node if either input requires grad
    if (requires_grad() || other.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<ops::MulNode<T>>(*this, other, result);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return result;
}

// Template instantiations for commonly used types
template class Tensor<float>;
template class Tensor<double>;

} // namespace dl
