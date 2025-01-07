#include "dl/tensor.hpp"
#include "dl/autograd.hpp"
#include "dl/ops/broadcast.hpp"
#include "dl/ops/basic_ops.hpp"
#include <stdexcept>
#include <numeric>

namespace dl {

template<typename T>
Tensor<T>::TensorImpl::TensorImpl(const std::vector<size_t>& shape)
    : shape_(shape)
    , data_(std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>()))
    , grad_(data_.size(), T(0))
    , requires_grad_(false) {}

template<typename T>
const std::vector<T>& Tensor<T>::TensorImpl::data() const { return data_; }

template<typename T>
std::vector<T>& Tensor<T>::TensorImpl::data() { return data_; }

template<typename T>
const std::vector<T>& Tensor<T>::TensorImpl::grad() const { return grad_; }

template<typename T>
std::vector<T>& Tensor<T>::TensorImpl::grad() { return grad_; }

template<typename T>
const std::vector<size_t>& Tensor<T>::TensorImpl::shape() const { return shape_; }

template<typename T>
bool Tensor<T>::TensorImpl::requires_grad() const { return requires_grad_; }

template<typename T>
void Tensor<T>::TensorImpl::set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

template<typename T>
void Tensor<T>::TensorImpl::zero_grad() {
    std::fill(grad_.begin(), grad_.end(), T(0));
}

template<typename T>
Tensor<T>::Tensor() : impl_(std::make_shared<TensorImpl>(std::vector<size_t>{0})) {}

template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape) : impl_(std::make_shared<TensorImpl>(shape)) {}

template<typename T>
const std::vector<size_t>& Tensor<T>::shape() const { return impl_->shape(); }

template<typename T>
const std::vector<T>& Tensor<T>::data() const { return impl_->data(); }

template<typename T>
std::vector<T>& Tensor<T>::data() { return impl_->data(); }

template<typename T>
const std::vector<T>& Tensor<T>::grad() const { return impl_->grad(); }

template<typename T>
std::vector<T>& Tensor<T>::grad() { return impl_->grad(); }

template<typename T>
bool Tensor<T>::requires_grad() const { return impl_->requires_grad(); }

template<typename T>
void Tensor<T>::set_requires_grad(bool requires_grad) { impl_->set_requires_grad(requires_grad); }

template<typename T>
void Tensor<T>::backward() {
    if (!impl_) {
        throw std::runtime_error("Tensor implementation is null");
    }
    if (!requires_grad()) {
        return;
    }
    
    // Set initial gradient
    auto& grad_data = grad();
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = T(1);
    }
    
    // Run backward pass through computation graph
    ComputationGraph::getInstance().backward();
    ComputationGraph::getInstance().clear();
}

template<typename T>
void Tensor<T>::zero_grad() {
    if (!impl_) {
        throw std::runtime_error("Tensor implementation is null");
    }
    impl_->zero_grad();
}

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
    
    if (requires_grad() || other.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<ops::MulNode<T>>(*this, other, result);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return result;
}

// Explicit template instantiation for float
template class Tensor<float>;

} // namespace dl
