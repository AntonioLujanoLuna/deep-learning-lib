#pragma once

#include "tensor_error.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

namespace dl {
namespace detail {

template<typename T>
class TensorImpl {
public:
    using Shape = std::vector<size_t>;
    using Data = std::vector<T>;

    explicit TensorImpl(const Shape& shape)
        : shape_(shape)
        , requires_grad_(false) {
        validate_shape(shape);
        allocate_data();
    }

    TensorImpl(const Shape& shape, const Data& data)
        : shape_(shape)
        , data_(data)
        , requires_grad_(false) {
        validate_shape(shape);
        validate_data_size();
    }

    // Move constructors
    TensorImpl(TensorImpl&&) noexcept = default;
    TensorImpl& operator=(TensorImpl&&) noexcept = default;

    // Copy constructors
    TensorImpl(const TensorImpl&) = default;
    TensorImpl& operator=(const TensorImpl&) = default;

    // Core accessors
    const Shape& shape() const noexcept { return shape_; }
    Data& data() noexcept { return data_; }
    const Data& data() const noexcept { return data_; }
    
    // Gradient operations
    Data& grad() { 
        check_requires_grad();
        ensure_grad_initialized();
        return grad_; 
    }
    
    const Data& grad() const { 
        check_requires_grad();
        return grad_; 
    }

    bool requires_grad() const noexcept { return requires_grad_; }
    
    void set_requires_grad(bool requires_grad) { 
        requires_grad_ = requires_grad; 
        if (requires_grad) {
            ensure_grad_initialized();
        }
    }

    void zero_grad() {
        if (requires_grad_) {
            std::fill(grad_.begin(), grad_.end(), T(0));
        }
    }

    // Utility functions
    size_t num_elements() const noexcept {
        return data_.size();
    }

    size_t num_dimensions() const noexcept {
        return shape_.size();
    }

private:
    Shape shape_;
    Data data_;
    Data grad_;
    bool requires_grad_;

    void validate_shape(const Shape& shape) {
        if (shape.empty()) {
            throw TensorError("Empty shape is not allowed");
        }
        for (size_t dim : shape) {
            if (dim == 0) {
                throw TensorError("Zero dimension is not allowed");
            }
        }
    }

    void validate_data_size() {
        size_t expected_size = compute_size();
        if (data_.size() != expected_size) {
            throw TensorError("Data size does not match shape");
        }
    }

    size_t compute_size() const {
        return std::accumulate(shape_.begin(), shape_.end(), 
                             size_t(1), std::multiplies<size_t>());
    }

    void allocate_data() {
        data_.resize(compute_size());
    }

    void ensure_grad_initialized() {
        if (grad_.empty()) {
            grad_.resize(data_.size(), T(0));
        }
    }

    void check_requires_grad() const {
        if (!requires_grad_) {
            throw TensorError("Tensor does not require gradients");
        }
    }
};

} // namespace detail
} // namespace dl
