#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <numeric>

namespace dl {

template<typename T>
class Tensor;  // Forward declaration

namespace detail {
    template<typename T>
    class TensorImpl {
    public:
        TensorImpl(const std::vector<size_t>& shape) 
            : shape_(shape)
            , requires_grad_(false) {
            // Validate shape
            if (shape.empty()) {
                throw std::runtime_error("Cannot create tensor with empty shape");
            }
            
            // Compute total size
            size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
            if (total_size == 0) {
                throw std::runtime_error("Cannot create tensor with zero size");
            }
            
            // Initialize data and grad with correct size
            data_.resize(total_size, T(0));
            grad_.resize(total_size, T(0));
            
            std::cout << "TensorImpl created with shape: [";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << shape[i];
            }
            std::cout << "], size: " << total_size << std::endl;
        }

        const std::vector<T>& data() const { return data_; }
        std::vector<T>& data() { return data_; }
        
        const std::vector<T>& grad() const { 
            if (!requires_grad_) {
                std::cerr << "Warning: Accessing gradient of tensor that doesn't require gradients" << std::endl;
            }
            return grad_; 
        }
        
        std::vector<T>& grad() { 
            if (!requires_grad_) {
                std::cerr << "Warning: Accessing gradient of tensor that doesn't require gradients" << std::endl;
            }
            return grad_; 
        }
        
        const std::vector<size_t>& shape() const { return shape_; }
        bool requires_grad() const { return requires_grad_; }
        
        void set_requires_grad(bool requires_grad) { 
            std::cout << "Setting requires_grad to " << (requires_grad ? "true" : "false") << std::endl;
            std::cout << "Current grad size: " << grad_.size() << std::endl;
            
            // Store current gradients if we're enabling requires_grad
            std::vector<T> temp_grad;
            if (requires_grad && !requires_grad_) {
                temp_grad = grad_;
            }
            
            requires_grad_ = requires_grad;
            
            // Restore gradients if we enabled requires_grad
            if (requires_grad && !temp_grad.empty()) {
                grad_ = temp_grad;
            }
            // Only zero gradients when disabling requires_grad
            else if (!requires_grad) {
                zero_grad();
            }
            
            std::cout << "New grad size: " << grad_.size() << std::endl;
        }
        
        void zero_grad() { 
            if (requires_grad_) {
                std::fill(grad_.begin(), grad_.end(), T(0));
            }
        }

    private:
        std::vector<size_t> shape_;
        std::vector<T> data_;
        std::vector<T> grad_;
        bool requires_grad_;
    };
} // namespace detail

template<typename T>
class Tensor {
public:
    using TensorPtr = std::shared_ptr<detail::TensorImpl<T>>;

    Tensor() : impl_(std::make_shared<detail::TensorImpl<T>>(std::vector<size_t>{1, 1})) {}
    
    explicit Tensor(const std::vector<size_t>& shape) 
        : impl_(std::make_shared<detail::TensorImpl<T>>(shape.empty() ? std::vector<size_t>{1, 1} : shape)) {}
    
    // Copy constructor
    Tensor(const Tensor& other) : impl_(other.impl_) {
        if (!impl_) {
            throw std::runtime_error("Cannot copy tensor with null implementation");
        }
    }
    
    // Assignment operator
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (!other.impl_) {
                throw std::runtime_error("Cannot assign from tensor with null implementation");
            }
            impl_ = other.impl_;
        }
        return *this;
    }
    
    const std::vector<size_t>& shape() const { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        return impl_->shape(); 
    }
    
    const std::vector<T>& data() const { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        return impl_->data(); 
    }
    
    std::vector<T>& data() { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        return impl_->data(); 
    }
    
    const std::vector<T>& grad() const { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        return impl_->grad(); 
    }
    
    std::vector<T>& grad() { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        return impl_->grad(); 
    }
    
    bool requires_grad() const { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        return impl_->requires_grad(); 
    }
    
    void set_requires_grad(bool requires_grad) { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        impl_->set_requires_grad(requires_grad); 
    }
    
    void zero_grad() { 
        if (!impl_) {
            throw std::runtime_error("Tensor implementation is null");
        }
        impl_->zero_grad(); 
    }

    // Operators
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;

private:
    TensorPtr impl_;
};

} // namespace dl

// Include these after the Tensor class definition
#include "autograd.hpp"
#include "ops/broadcast.hpp"
#include "ops/basic_ops.hpp"

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

} // namespace dl
