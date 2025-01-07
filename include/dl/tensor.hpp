#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

namespace dl {
namespace ops {

template<typename T>
class AddNode;

template<typename T>
class MulNode;

} // namespace ops

template<typename T = float>
class Tensor {
public:
    // Implementation class
    class TensorImpl {
    public:
        TensorImpl(const std::vector<size_t>& shape);
        const std::vector<T>& data() const;
        std::vector<T>& data();
        const std::vector<T>& grad() const;
        std::vector<T>& grad();
        const std::vector<size_t>& shape() const;
        bool requires_grad() const;
        void set_requires_grad(bool requires_grad);
        void zero_grad();

    private:
        std::vector<size_t> shape_;
        std::vector<T> data_;
        std::vector<T> grad_;
        bool requires_grad_;
    };

    using TensorPtr = std::shared_ptr<TensorImpl>;

    // Constructor
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape);
    
    // Copy constructor and assignment
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    
    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    
    // Accessors
    const std::vector<size_t>& shape() const;
    const std::vector<T>& data() const;
    std::vector<T>& data();
    const std::vector<T>& grad() const;
    std::vector<T>& grad();
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    
    // Gradient operations
    void backward();
    void zero_grad();

    // Basic operations
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;

private:
    TensorPtr impl_;
};

} // namespace dl
