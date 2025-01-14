#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include "../utils/utils.hpp"
#include <memory>
#include <vector>
#include <utility>

namespace dl {
namespace ops {

template<typename T>
class SliceNode : public Node {
public:
    SliceNode(const Tensor<T>& input, 
              const std::vector<std::pair<size_t, size_t>>& ranges,
              Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , ranges_(ranges) {
        // Store input shape for backward pass
        input_shape_ = input.shape();
    }

    std::string node_type() const override { return "Slice"; }
    
    void backward() override {
        if (!input_impl_->requires_grad()) return;
        
        std::vector<T>& input_grad = input_impl_->grad();
        const std::vector<T>& output_grad = output_impl_->grad();
        
        // Calculate strides for input tensor
        std::vector<size_t> strides(input_shape_.size());
        strides.back() = 1;
        for (int i = input_shape_.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * input_shape_[i + 1];
        }
        
        // Map output indices back to input indices and accumulate gradients
        size_t output_idx = 0;
        std::vector<size_t> curr_indices(ranges_.size());
        
        while (output_idx < output_grad.size()) {
            // Calculate input index from current indices
            size_t input_idx = 0;
            for (size_t dim = 0; dim < ranges_.size(); ++dim) {
                input_idx += (ranges_[dim].first + curr_indices[dim]) * strides[dim];
            }
            
            // Accumulate gradient
            input_grad[input_idx] += output_grad[output_idx];
            
            // Update indices
            for (int dim = ranges_.size() - 1; dim >= 0; --dim) {
                curr_indices[dim]++;
                if (curr_indices[dim] < (ranges_[dim].second - ranges_[dim].first)) {
                    break;
                }
                curr_indices[dim] = 0;
            }
            output_idx++;
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    std::vector<std::pair<size_t, size_t>> ranges_;
    std::vector<size_t> input_shape_;
};

// Helper function to compute output shape from ranges
inline std::vector<size_t> compute_slice_shape(
    const std::vector<size_t>& input_shape,
    const std::vector<std::pair<size_t, size_t>>& ranges) {
    
    std::vector<size_t> output_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        output_shape.push_back(ranges[i].second - ranges[i].first);
    }
    return output_shape;
}

// Helper function to compute linear index from multi-dimensional indices
inline size_t compute_linear_index(
    const std::vector<size_t>& indices,
    const std::vector<size_t>& shape) {
    
    size_t index = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

} // namespace ops
} // namespace dl
