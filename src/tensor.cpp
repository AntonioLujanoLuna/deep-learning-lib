#include "dl/tensor.hpp"
#include "dl/ops/basic_ops.hpp"
#include "dl/ops/broadcast.hpp"
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

// Explicit template instantiations
template class Tensor<float>;
template class Tensor<double>;

} // namespace dl