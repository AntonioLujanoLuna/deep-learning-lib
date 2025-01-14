#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include "../utils/utils.hpp"
#include <memory>
#include <vector>
#include <limits>
#include <numeric>

namespace dl {
namespace ops {

template<typename T>
class SumNode : public Node {
public:
    SumNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , dim_(dim) {}

    std::string node_type() const override { return "Sum"; }
    
    void backward() override {
        if (!input_impl_->requires_grad()) return;
        
        // Broadcast gradients back to input shape
        std::vector<T>& input_grad = input_impl_->grad();
        const std::vector<T>& output_grad = output_impl_->grad();
        
        // For each element in input_grad, add the corresponding output_grad
        // Implementation depends on dimension being reduced
        // This is a simplified version for the entire tensor
        for (size_t i = 0; i < input_grad.size(); ++i) {
            input_grad[i] += output_grad[0];
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    int dim_;
};

template<typename T>
class MeanNode : public Node {
public:
    MeanNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , dim_(dim)
        , scale_(1.0 / input.shape()[dim]) {}

    std::string node_type() const override { return "Mean"; }
    
    void backward() override {
        if (!input_impl_->requires_grad()) return;
        
        std::vector<T>& input_grad = input_impl_->grad();
        const std::vector<T>& output_grad = output_impl_->grad();
        
        // Scale the gradients by 1/N
        for (size_t i = 0; i < input_grad.size(); ++i) {
            input_grad[i] += output_grad[0] * scale_;
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    int dim_;
    T scale_;
};

template<typename T>
class MaxNode : public Node {
public:
    MaxNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , dim_(dim) {
        // Store indices of maximum elements for backward pass
        const std::vector<T>& input_data = input_impl_->data();
        max_indices_.resize(output.data().size());
        // Implementation depends on dimension being reduced
        // This is a simplified version
        size_t max_idx = 0;
        T max_val = input_data[0];
        for (size_t i = 1; i < input_data.size(); ++i) {
            if (input_data[i] > max_val) {
                max_val = input_data[i];
                max_idx = i;
            }
        }
        max_indices_[0] = max_idx;
    }

    std::string node_type() const override { return "Max"; }
    
    void backward() override {
        if (!input_impl_->requires_grad()) return;
        
        std::vector<T>& input_grad = input_impl_->grad();
        const std::vector<T>& output_grad = output_impl_->grad();
        
        // Only propagate gradients to the maximum elements
        for (size_t i = 0; i < max_indices_.size(); ++i) {
            input_grad[max_indices_[i]] += output_grad[i];
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    int dim_;
    std::vector<size_t> max_indices_;
};

// Free functions for creating reduction operations
template<typename T>
Tensor<T> sum(const Tensor<T>& input, int dim = -1) {
    // -1 means reduce all dimensions
    std::vector<size_t> output_shape;
    if (dim == -1) {
        output_shape = {1};  // Scalar output
    } else {
        output_shape = input.shape();
        output_shape[dim] = 1;
    }
    
    Tensor<T> output(output_shape);
    if (input.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<SumNode<T>>(input, dim, output);
        ComputationGraph::getInstance().add_node(node);
    }
    
    // Compute the sum
    const std::vector<T>& input_data = input.data();
    std::vector<T>& output_data = output.data();
    output_data[0] = std::accumulate(input_data.begin(), input_data.end(), T(0));
    
    return output;
}

template<typename T>
Tensor<T> mean(const Tensor<T>& input, int dim = -1) {
    // Similar to sum, but divide by number of elements
    auto result = sum(input, dim);
    auto num_elements = input.data().size();
    std::vector<T>& output_data = result.data();
    for (auto& val : output_data) {
        val /= static_cast<T>(num_elements);
    }
    return result;
}

template<typename T>
Tensor<T> max(const Tensor<T>& input, int dim = -1) {
    std::vector<size_t> output_shape;
    if (dim == -1) {
        output_shape = {1};
    } else {
        output_shape = input.shape();
        output_shape[dim] = 1;
    }
    
    Tensor<T> output(output_shape);
    if (input.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<MaxNode<T>>(input, dim, output);
        ComputationGraph::getInstance().add_node(node);
    }
    
    // Compute the max
    const std::vector<T>& input_data = input.data();
    std::vector<T>& output_data = output.data();
    output_data[0] = *std::max_element(input_data.begin(), input_data.end());
    
    return output;
}

} // namespace ops
} // namespace dl