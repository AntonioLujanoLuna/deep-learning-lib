#pragma once

#include "detail/tensor_impl.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace dl {

// Forward declarations
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

    // Operators
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;

private:
    TensorPtr impl_;

    // Allow nodes to access impl_
    template<typename U> friend class ops::AddNode;
    template<typename U> friend class ops::MulNode;
    template<typename U> friend class ops::MatMulNode;
    template<typename U> friend class ops::ReLUNode;
    template<typename U> friend class ops::SigmoidNode;
    template<typename U> friend class ops::TanhNode;
    template<typename U> friend class nn::LinearNode;
};

} // namespace dl
