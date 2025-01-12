#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

namespace dl {
namespace detail {

template<typename T>
class TensorImpl {
public:
    TensorImpl(const std::vector<size_t>& shape)
        : shape_(shape)
        , requires_grad_(false) {
        size_t size = 1;
        for (size_t dim : shape) {
            size *= dim;
        }
        data_.resize(size);
    }

    TensorImpl(const std::vector<size_t>& shape, const std::vector<T>& data)
        : shape_(shape)
        , data_(data)
        , requires_grad_(false) {
        size_t expected_size = 1;
        for (size_t dim : shape) {
            expected_size *= dim;
        }
        if (data.size() != expected_size) {
            throw std::runtime_error("Data size does not match shape");
        }
    }

    const std::vector<size_t>& shape() const { return shape_; }
    std::vector<T>& data() { return data_; }
    const std::vector<T>& data() const { return data_; }
    std::vector<T>& grad() { 
        if (!requires_grad_) {
            throw std::runtime_error("Tensor does not require gradients");
        }
        if (grad_.empty()) {
            grad_.resize(data_.size(), T(0));
        }
        return grad_; 
    }
    const std::vector<T>& grad() const { 
        if (!requires_grad_) {
            throw std::runtime_error("Tensor does not require gradients");
        }
        return grad_; 
    }

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { 
        requires_grad_ = requires_grad; 
        if (requires_grad) {
            grad_.resize(data_.size(), T(0));
        }
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
} // namespace dl
