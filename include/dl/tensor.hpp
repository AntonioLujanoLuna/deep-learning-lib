#pragma once

#include "fwd.hpp"
#include "detail/tensor_impl.hpp"
#include "dl/ops/broadcast.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace dl {

namespace nn {
template<typename T> class LinearNode;
}

template<typename T>
class Tensor {
public:
    using TensorPtr = std::shared_ptr<detail::TensorImpl<T>>;

    // Constructors remain the same since they create new tensors
    Tensor(const std::vector<size_t>& shape)
        : impl_(std::make_shared<detail::TensorImpl<T>>(shape)) {}

    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
        : impl_(std::make_shared<detail::TensorImpl<T>>(shape, data)) {}

    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    // Core accessors remain unchanged
    const std::vector<size_t>& shape() const { return impl_->shape(); }
    std::vector<T>& data() { return impl_->data(); }
    const std::vector<T>& data() const { return impl_->data(); }
    std::vector<T>& grad() { return impl_->grad(); }
    const std::vector<T>& grad() const { return impl_->grad(); }

    bool requires_grad() const { return impl_->requires_grad(); }
    void set_requires_grad(bool requires_grad) { impl_->set_requires_grad(requires_grad); }
    void zero_grad() { impl_->zero_grad(); }

    void setGradFn(const std::weak_ptr<Node>& node) {
        impl_->grad_fn_ = node;
    }

    std::weak_ptr<Node> gradFn() const {
        return impl_->grad_fn_;
    }

    // Updated operator declarations to return shared_ptr
    std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& other) const {
        if (!impl_ || !other.impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }

        auto broadcast_shape = ops::compute_broadcast_shape(shape(), other.shape());
        
        // Broadcast inputs if necessary
        std::vector<T> a_data = (shape() == broadcast_shape) ? 
            data() : ops::broadcast_to(data(), shape(), broadcast_shape);
        std::vector<T> b_data = (other.shape() == broadcast_shape) ? 
            other.data() : ops::broadcast_to(other.data(), other.shape(), broadcast_shape);
        
        // Create result as shared_ptr
        auto result = std::make_shared<Tensor<T>>(broadcast_shape);
        auto& result_data = result->data();
        
        for (size_t i = 0; i < result_data.size(); ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
        
        if (requires_grad() || other.requires_grad()) {
            result->set_requires_grad(true);
            // Create the node and set it as gradient function
            auto node = std::make_shared<ops::AddNode<T>>(*this, other, *result);
            result->setGradFn(node);
            ComputationGraph::getInstance().addNode(node);
        }
        
        return result;
    }

    std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& other) const {
        if (!impl_ || !other.impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }

        auto broadcast_shape = ops::compute_broadcast_shape(shape(), other.shape());
        
        std::vector<T> a_data = (shape() == broadcast_shape) ? 
            data() : ops::broadcast_to(data(), shape(), broadcast_shape);
        std::vector<T> b_data = (other.shape() == broadcast_shape) ? 
            other.data() : ops::broadcast_to(other.data(), other.shape(), broadcast_shape);
        
        auto result = std::make_shared<Tensor<T>>(broadcast_shape);
        auto& result_data = result->data();
        
        for (size_t i = 0; i < result_data.size(); ++i) {
            result_data[i] = a_data[i] * b_data[i];
        }
        
        if (requires_grad() || other.requires_grad()) {
            result->set_requires_grad(true);
            auto node = std::make_shared<ops::MulNode<T>>(*this, other, *result);
            result->setGradFn(node);
            ComputationGraph::getInstance().addNode(node);
        }
        
        return result;
    }

    // Operator overloads for shared_ptr inputs
    std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& other) const {
        return *this + *other;
    }
    
    std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& other) const {
        return *this + *other;
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> impl_;

    // Friend declarations for operation nodes
    template<typename U> friend class ops::AddNode;
    template<typename U> friend class ops::MulNode;
    template<typename U> friend class ops::MatMulNode;
    template<typename U> friend class ops::ReLUNode;
    template<typename U> friend class ops::SigmoidNode;
    template<typename U> friend class ops::TanhNode;
    template<typename U> friend class nn::LinearNode;
};

// Free function implementations
template<typename T>
std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& a, const Tensor<T>& b) {
    return (*a) + b;
}

template<typename T>
std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& a, const Tensor<T>& b) {
    return (*a) * b;
}

template<typename T>
std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& a, 
                                    const std::shared_ptr<Tensor<T>>& b) {
    return (*a) + (*b);
}

template<typename T>
std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& a, 
                                    const std::shared_ptr<Tensor<T>>& b) {
    return (*a) * (*b);
}

// Template instantiations
template class Tensor<float>;
template class Tensor<double>;

// Explicit instantiations for operator overloads
template std::shared_ptr<Tensor<float>> operator+(const std::shared_ptr<Tensor<float>>&, 
                                                 const Tensor<float>&);
template std::shared_ptr<Tensor<float>> operator*(const std::shared_ptr<Tensor<float>>&, 
                                                 const Tensor<float>&);
template std::shared_ptr<Tensor<float>> operator+(const std::shared_ptr<Tensor<float>>&, 
                                                 const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<float>> operator*(const std::shared_ptr<Tensor<float>>&, 
                                                 const std::shared_ptr<Tensor<float>>&);

// Similar instantiations for double
template std::shared_ptr<Tensor<double>> operator+(const std::shared_ptr<Tensor<double>>&, 
                                                  const Tensor<double>&);
template std::shared_ptr<Tensor<double>> operator*(const std::shared_ptr<Tensor<double>>&, 
                                                  const Tensor<double>&);
template std::shared_ptr<Tensor<double>> operator+(const std::shared_ptr<Tensor<double>>&, 
                                                  const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<double>> operator*(const std::shared_ptr<Tensor<double>>&, 
                                                  const std::shared_ptr<Tensor<double>>&);

} // namespace dl