#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "utils.hpp"

namespace dl {

// Forward declarations
template<typename T>
class Tensor;

// Utility functions
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
            
            // Compute total size and validate
            size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
            if (total_size == 0) {
                throw std::runtime_error("Cannot create tensor with zero size");
            }
            
            // Initialize data and grad with correct size
            data_.resize(total_size, T(0));
            grad_.resize(total_size, T(0));
            
            // Ensure grad vector is properly initialized
            if (requires_grad_) {
                std::fill(grad_.begin(), grad_.end(), T(0));
            }
        }

        const std::vector<T>& data() const { return data_; }
        std::vector<T>& data() { return data_; }
        
        const std::vector<T>& grad() const { 
            if (shape_.empty()) {
                throw std::runtime_error("Cannot access gradient: tensor has empty shape");
            }
            if (!requires_grad_) {
                throw std::runtime_error("Cannot access gradient: tensor does not require gradients");
            }
            if (grad_.empty()) {
                throw std::runtime_error("Cannot access gradient: gradient vector is empty");
            }
            return grad_; 
        }
        
        std::vector<T>& grad() { 
            if (shape_.empty()) {
                throw std::runtime_error("Cannot access gradient: tensor has empty shape");
            }
            if (!requires_grad_) {
                throw std::runtime_error("Cannot access gradient: tensor does not require gradients");
            }
            
            // Ensure gradient vector has correct size
            size_t total_size = std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
            if (grad_.size() != total_size) {
                grad_.resize(total_size, T(0));
            }
            
            return grad_; 
        }
        
        const std::vector<size_t>& shape() const { 
            if (shape_.empty()) {
                throw std::runtime_error("Tensor has empty shape");
            }
            return shape_; 
        }
        
        void set_requires_grad(bool requires_grad) {
            requires_grad_ = requires_grad;
            
            // Handle gradient initialization/deinitialization
            if (requires_grad) {
                size_t total_size = std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
                grad_.resize(total_size, T(0));
            } else {
                grad_.clear();  // Free memory if gradients are no longer needed
            }
        }

        bool requires_grad() const { 
            return requires_grad_; 
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

    Tensor() : impl_(std::make_shared<detail::TensorImpl<T>>(std::vector<size_t>{1, 1})) {
        //std::cout << "Creating default tensor with shape [1, 1] (impl=" << impl_.get() << ")" << std::endl;
    }

    explicit Tensor(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            //std::cout << "Creating tensor with default shape [1, 1] (empty shape provided)" << std::endl;
            impl_ = std::make_shared<detail::TensorImpl<T>>(std::vector<size_t>{1, 1});
        } else {
            //std::cout << "Creating tensor with shape " << utils::shape_to_string(shape) << std::endl;
            impl_ = std::make_shared<detail::TensorImpl<T>>(shape);
        }
        //std::cout << "  impl=" << impl_.get() << std::endl;
    }

    // Copy constructor - create new implementation
    Tensor(const Tensor& other) {
        if (!other.impl_) {
            throw std::runtime_error("Cannot copy tensor with null implementation");
        }
        impl_ = std::make_shared<detail::TensorImpl<T>>(other.impl_->shape());
        impl_->data() = other.impl_->data();
        impl_->set_requires_grad(other.impl_->requires_grad());
        
        // Initialize gradient to zero, don't copy from other tensor
        if (impl_->requires_grad()) {
            size_t total_size = std::accumulate(impl_->shape().begin(), impl_->shape().end(), size_t(1), std::multiplies<size_t>());
            impl_->grad().resize(total_size, T(0));
        }
    }
    
    // Assignment operator - create new implementation
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (!other.impl_) {
                throw std::runtime_error("Cannot assign from tensor with null implementation");
            }
            impl_ = std::make_shared<detail::TensorImpl<T>>(other.impl_->shape());
            impl_->data() = other.impl_->data();
            impl_->set_requires_grad(other.impl_->requires_grad());
            
            // Initialize gradient to zero, don't copy from other tensor
            if (impl_->requires_grad()) {
                size_t total_size = std::accumulate(impl_->shape().begin(), impl_->shape().end(), size_t(1), std::multiplies<size_t>());
                impl_->grad().resize(total_size, T(0));
            }
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

#include <iostream>
#include <numeric>
#include <sstream>
#include <iomanip>

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

} // namespace dl
