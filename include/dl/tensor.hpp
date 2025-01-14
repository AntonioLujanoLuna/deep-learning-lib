#pragma once

#include "detail/tensor_impl.hpp"
#include <memory>
#include <vector>
#include <utility>

namespace dl {

// Forward declarations
namespace ops {
template<typename T> class AddNode;
template<typename T> class MulNode;
template<typename T> class MatMulNode;
template<typename T> class ReLUNode;
template<typename T> class SigmoidNode;
template<typename T> class TanhNode;
template<typename T> class SumNode;
template<typename T> class MeanNode;
template<typename T> class MaxNode;
template<typename T> class SliceNode;
}

template<typename T>
class Tensor {
public:
    // Type aliases for clarity
    using TensorPtr = std::shared_ptr<detail::TensorImpl<T>>;
    using Shape = std::vector<size_t>;
    using Data = std::vector<T>;
    using Range = std::pair<size_t, size_t>;

    // Constructors
    explicit Tensor(const Shape& shape)
        : impl_(std::make_shared<detail::TensorImpl<T>>(shape)) {}

    Tensor(const Shape& shape, const Data& data)
        : impl_(std::make_shared<detail::TensorImpl<T>>(shape, data)) {}
    
    // Move constructors and assignment
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    
    // Copy constructors and assignment
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    // Core accessors
    const Shape& shape() const noexcept { return impl_->shape(); }
    Data& data() { return impl_->data(); }
    const Data& data() const { return impl_->data(); }
    
    // Gradient operations
    Data& grad() { return impl_->grad(); }
    const Data& grad() const { return impl_->grad(); }
    bool requires_grad() const noexcept { return impl_->requires_grad(); }
    void set_requires_grad(bool requires_grad) { impl_->set_requires_grad(requires_grad); }
    void zero_grad() { impl_->zero_grad(); }

    // Utility functions
    size_t num_elements() const noexcept { return impl_->num_elements(); }
    size_t num_dimensions() const noexcept { return impl_->num_dimensions(); }

    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    // Reduction operations
    Tensor sum(int dim = -1) const;
    Tensor mean(int dim = -1) const;
    Tensor max(int dim = -1) const;

    // Slicing operations
    Tensor slice(const std::vector<Range>& ranges) const;
    Tensor operator[](size_t index) const;

protected:
    // Protected constructor for operations
    explicit Tensor(TensorPtr impl) : impl_(std::move(impl)) {}

private:
    TensorPtr impl_;

    // Friend declarations only for core operation classes
    template<typename U> friend class ops::AddNode;
    template<typename U> friend class ops::MulNode;
    template<typename U> friend class ops::MatMulNode;
    template<typename U> friend class ops::ReLUNode;
    template<typename U> friend class ops::SigmoidNode;
    template<typename U> friend class ops::TanhNode;
    template<typename U> friend class ops::SumNode;
    template<typename U> friend class ops::MeanNode;
    template<typename U> friend class ops::MaxNode;
    template<typename U> friend class ops::SliceNode;
};

// Template instantiations for commonly used types
template class Tensor<float>;
template class Tensor<double>;

} // namespace dl

// Include operation implementations
#include "ops/basic_ops.hpp"
#include "ops/matrix_ops.hpp"
#include "ops/reduction_ops.hpp"
#include "ops/slice_ops.hpp"
