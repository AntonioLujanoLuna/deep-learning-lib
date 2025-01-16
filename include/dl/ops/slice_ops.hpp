#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include "../utils/utils.hpp"
#include "../utils/index_utils.hpp"
#include <memory>
#include <vector>
#include <utility>

namespace dl {
namespace ops {

template<typename T>
class SliceNode 
    : public Node
    , public std::enable_shared_from_this<SliceNode<T>> {
public:
    SliceNode(const Tensor<T>& input, 
              const std::vector<std::pair<size_t, size_t>>& ranges,
              Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , ranges_(ranges) 
    {
        // Store input shape for backward pass
        input_shape_ = input.shape();

        // -------------------------
        // DAG linking
        // -------------------------
        // 1) Output is produced by this SliceNode
        output.setGradFn(this->shared_from_this());

        // 2) If the input has a gradFn, link it as a parent
        if (auto parent = input.gradFn().lock()) {
            parents_.push_back(parent);
            parent->children_.push_back(this->shared_from_this());
        }

        // 3) Register this node in the ComputationGraph
        ComputationGraph::getInstance().addNode(this->shared_from_this());
    }

    std::string node_type() const override { return "Slice"; }
    
    void backward() override {
        if (!input_impl_->requires_grad()) return;
        
        std::vector<T>& input_grad = input_impl_->grad();
        const std::vector<T>& output_grad = output_impl_->grad();
        
        // Calculate strides for input tensor (row-major)
        std::vector<size_t> strides(input_shape_.size());
        strides.back() = 1;
        for (int i = static_cast<int>(input_shape_.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * input_shape_[i + 1];
        }
        
        // Map output indices back to input indices
        size_t output_idx = 0;
        std::vector<size_t> curr_indices(ranges_.size(), 0);

        // We'll iterate over each element in the sliced output
        while (output_idx < output_grad.size()) {
            // Calculate input index from curr_indices
            size_t input_idx = 0;
            for (size_t dim = 0; dim < ranges_.size(); ++dim) {
                input_idx += (ranges_[dim].first + curr_indices[dim]) * strides[dim];
            }
            
            // Accumulate gradient
            input_grad[input_idx] += output_grad[output_idx];
            
            // Increment the 'curr_indices' in a multi-dimensional manner
            for (int dim = static_cast<int>(ranges_.size()) - 1; dim >= 0; --dim) {
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

// ---------------------------------------------------------------------------
// Helper function to compute output shape from 'ranges'
// ---------------------------------------------------------------------------
inline std::vector<size_t> compute_slice_shape(
    const std::vector<size_t>& input_shape,
    const std::vector<std::pair<size_t, size_t>>& ranges) 
{
    std::vector<size_t> output_shape;
    output_shape.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        output_shape.push_back(ranges[i].second - ranges[i].first);
    }
    return output_shape;
}

// ---------------------------------------------------------------------------
// Helper function to compute linear index from multi-dimensional indices
// ---------------------------------------------------------------------------
inline size_t compute_linear_index(
    const std::vector<size_t>& indices,
    const std::vector<size_t>& shape) 
{
    size_t index = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

// ---------------------------------------------------------------------------
// (Optional) Free function to slice a Tensor (forward pass + node creation)
// ---------------------------------------------------------------------------
template<typename T>
Tensor<T> slice(const Tensor<T>& input,
                const std::vector<std::pair<size_t, size_t>>& ranges)
{
    // 1) Compute output shape
    auto out_shape = compute_slice_shape(input.shape(), ranges);

    // 2) Create output Tensor
    Tensor<T> output(out_shape);

    // 3) Perform the forward pass (copy the sliced data)
    //    We'll do naive iteration
    std::vector<T>& out_data = output.data();
    out_data.resize(1); // will fix up the final size below

    // We'll count how many elements the slice has
    size_t slice_size = 1;
    for (auto &r : ranges) {
        slice_size *= (r.second - r.first);
    }
    out_data.resize(slice_size);

    // We'll do a multi-dimensional iteration over all positions in the slice
    std::vector<size_t> curr_indices(ranges.size(), 0);
    size_t out_idx = 0;

    // We'll also build 'strides' for the input
    std::vector<size_t> strides(input.shape().size());
    strides.back() = 1;
    for (int i = static_cast<int>(input.shape().size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape()[i + 1];
    }

    while (true) {
        // Compute input_idx
        size_t input_idx = 0;
        for (size_t d = 0; d < ranges.size(); ++d) {
            input_idx += (ranges[d].first + curr_indices[d]) * strides[d];
        }

        // Copy the data
        out_data[out_idx] = input.data()[input_idx];
        out_idx++;

        // increment curr_indices
        int dim = static_cast<int>(ranges.size()) - 1;
        while (dim >= 0) {
            curr_indices[dim]++;
            if (curr_indices[dim] < (ranges[dim].second - ranges[dim].first)) {
                break;
            }
            curr_indices[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            // we've covered all positions
            break;
        }
    }

    // 4) If input requires grad, create a SliceNode
    if (input.requires_grad()) {
        auto node = std::make_shared<SliceNode<T>>(input, ranges, output);
        // The node constructor will handle setting output's gradFn, linking parents, etc.
    }

    return output;
}

} // namespace ops
} // namespace dl
