#pragma once

#include "../tensor.hpp"       // for Tensor<T>
#include "../autograd.hpp"     // for Node, ComputationGraph, etc.
#include "../utils/utils.hpp"  // or wherever you keep other necessary utils
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>

namespace dl {
namespace ops {

///////////////////////////////////////////////////////////
// (1) Utility functions for multi-dimensional index handling
///////////////////////////////////////////////////////////

// If you'd prefer to keep them in a separate file (e.g., index_utils.hpp),
// you can remove them here. But included for completeness:

inline std::vector<size_t> computeStrides(const std::vector<size_t>& shape) {
    // Row-major stride calculation
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

inline std::vector<size_t> unravelIndex(size_t idx, const std::vector<size_t>& shape) {
    // Convert a flattened index into multi-dimensional coordinates
    std::vector<size_t> coords(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t s = 1;
        for (size_t j = i + 1; j < shape.size(); ++j) {
            s *= shape[j];
        }
        coords[i] = (idx / s) % shape[i];
    }
    return coords;
}

inline size_t ravelIndex(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    // Convert multi-dimensional coordinates into a flattened index
    // using the provided row-major strides
    size_t idx = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        idx += coords[i] * strides[i];
    }
    return idx;
}

// A numerically stable accumulate (Kahan summation)
template<typename T>
inline T stable_accumulate(const std::vector<T>& values) {
    T sum = 0;
    T c = 0; // Compensation for lost low bits
    for (T v : values) {
        T y = v - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

///////////////////////////////////////////////////////////
// (2) Node Classes for Reduction Ops
///////////////////////////////////////////////////////////

/**
 * SumNode:
 *  - Sums values across a specified dimension (or all dims if dim = -1).
 */
template<typename T>
class SumNode : public Node {
public:
    SumNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , input_shape_(input.shape())
        , dim_(dim)
    {
        output_shape_ = output.shape();
    }

    std::string node_type() const override { return "Sum"; }

    void backward() override {
        if (!input_impl_->requires_grad()) return;

        auto& input_grad = input_impl_->grad();
        const auto& output_grad = output_impl_->grad();

        // If reducing all dims => single scalar
        if (dim_ == -1) {
            // Add output_grad[0] to each element of input
            T gval = output_grad[0];
            for (size_t i = 0; i < input_grad.size(); ++i) {
                input_grad[i] += gval;
            }
            return;
        }

        // Partial reduction along one dimension
        auto in_strides = computeStrides(input_shape_);
        auto out_strides = computeStrides(output_shape_);

        for (size_t i = 0; i < input_grad.size(); ++i) {
            // unravel the input index
            auto coords = unravelIndex(i, input_shape_);
            // collapsed dimension => coords[dim_] = 0 in output
            coords[dim_] = 0;
            size_t out_idx = ravelIndex(coords, out_strides);
            input_grad[i] += output_grad[out_idx];
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape_;
    int dim_;
};

/**
 * MeanNode:
 *  - Takes the mean across a specified dimension (or all dims if dim = -1).
 */
template<typename T>
class MeanNode : public Node {
public:
    MeanNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , input_shape_(input.shape())
        , dim_(dim)
    {
        // Figure out number of elements reduced
        if (dim_ == -1) {
            num_reduced_ = 1;
            for (auto s : input_shape_) {
                num_reduced_ *= s;
            }
        } else {
            num_reduced_ = input_shape_[dim_];
        }

        output_shape_ = output.shape();
        scale_ = T(1) / static_cast<T>(num_reduced_);
    }

    std::string node_type() const override { return "Mean"; }

    void backward() override {
        if (!input_impl_->requires_grad()) return;

        auto& input_grad = input_impl_->grad();
        const auto& output_grad = output_impl_->grad();

        if (dim_ == -1) {
            // All dims => single scalar
            T gval = output_grad[0] * scale_;
            for (size_t i = 0; i < input_grad.size(); ++i) {
                input_grad[i] += gval;
            }
            return;
        }

        // Partial dimension
        auto in_strides = computeStrides(input_shape_);
        auto out_strides = computeStrides(output_shape_);

        for (size_t i = 0; i < input_grad.size(); ++i) {
            auto coords = unravelIndex(i, input_shape_);
            coords[dim_] = 0; // collapsed dimension index
            size_t out_idx = ravelIndex(coords, out_strides);
            input_grad[i] += output_grad[out_idx] * scale_;
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape_;
    int dim_;
    size_t num_reduced_;
    T scale_;
};

/**
 * MaxNode:
 *  - Takes max across a dimension (or all dims), storing the index of the max
 *    for each slice to properly backprop only to max elements.
 */
template<typename T>
class MaxNode : public Node {
public:
    MaxNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , input_shape_(input.shape())
        , dim_(dim)
    {
        output_shape_ = output.shape();

        const auto& in_data = input_impl_->data();
        auto& out_data = output_impl_->data();

        if (dim_ == -1) {
            // Single scalar: global max
            size_t max_idx = 0;
            T max_val = in_data[0];
            for (size_t i = 1; i < in_data.size(); ++i) {
                if (in_data[i] > max_val) {
                    max_val = in_data[i];
                    max_idx = i;
                }
            }
            out_data.resize(1);
            out_data[0] = max_val;
            max_indices_.resize(1);
            max_indices_[0] = max_idx;
        } else {
            // Partial dimension
            size_t out_size = 1;
            for (auto s : output_shape_) out_size *= s;
            out_data.resize(out_size);
            max_indices_.resize(out_size);

            auto in_strides = computeStrides(input_shape_);
            auto out_strides = computeStrides(output_shape_);

            for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
                auto coords = unravelIndex(out_idx, output_shape_);

                T max_val = std::numeric_limits<T>::lowest();
                size_t max_pos = 0;

                // Vary the coordinate along dim_ from 0..(input_shape_[dim_]-1)
                for (size_t pos = 0; pos < input_shape_[dim_]; ++pos) {
                    coords[dim_] = pos;
                    size_t in_idx = ravelIndex(coords, in_strides);
                    if (in_data[in_idx] > max_val) {
                        max_val = in_data[in_idx];
                        max_pos = in_idx;
                    }
                }
                out_data[out_idx] = max_val;
                max_indices_[out_idx] = max_pos;
            }
        }
    }

    std::string node_type() const override { return "Max"; }

    void backward() override {
        if (!input_impl_->requires_grad()) return;

        auto& in_grad = input_impl_->grad();
        const auto& out_grad = output_impl_->grad();

        // Distribute gradient only to the max index
        for (size_t i = 0; i < max_indices_.size(); ++i) {
            in_grad[max_indices_[i]] += out_grad[i];
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape_;
    int dim_;
    std::vector<size_t> max_indices_;
};

/**
 * MinNode:
 *  - Similar to MaxNode, but picks the minimum value. 
 */
template<typename T>
class MinNode : public Node {
public:
    MinNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , input_shape_(input.shape())
        , dim_(dim)
    {
        output_shape_ = output.shape();

        const auto& in_data = input_impl_->data();
        auto& out_data = output_impl_->data();

        if (dim_ == -1) {
            // Single scalar: global min
            size_t min_idx = 0;
            T min_val = in_data[0];
            for (size_t i = 1; i < in_data.size(); ++i) {
                if (in_data[i] < min_val) {
                    min_val = in_data[i];
                    min_idx = i;
                }
            }
            out_data.resize(1);
            out_data[0] = min_val;
            min_indices_.resize(1);
            min_indices_[0] = min_idx;
        } else {
            // Partial dimension
            size_t out_size = 1;
            for (auto s : output_shape_) out_size *= s;
            out_data.resize(out_size);
            min_indices_.resize(out_size);

            auto in_strides = computeStrides(input_shape_);
            auto out_strides = computeStrides(output_shape_);

            for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
                auto coords = unravelIndex(out_idx, output_shape_);

                T min_val = std::numeric_limits<T>::max();
                size_t min_pos = 0;

                // Vary the coordinate along dim_
                for (size_t pos = 0; pos < input_shape_[dim_]; ++pos) {
                    coords[dim_] = pos;
                    size_t in_idx = ravelIndex(coords, in_strides);
                    if (in_data[in_idx] < min_val) {
                        min_val = in_data[in_idx];
                        min_pos = in_idx;
                    }
                }
                out_data[out_idx] = min_val;
                min_indices_[out_idx] = min_pos;
            }
        }
    }

    std::string node_type() const override { return "Min"; }

    void backward() override {
        if (!input_impl_->requires_grad()) return;

        auto& in_grad = input_impl_->grad();
        const auto& out_grad = output_impl_->grad();

        // Distribute gradient only to the min index
        for (size_t i = 0; i < min_indices_.size(); ++i) {
            in_grad[min_indices_[i]] += out_grad[i];
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape_;
    int dim_;
    std::vector<size_t> min_indices_;
};

/**
 * ProdNode:
 *  - Product of elements along a dimension (or all dims).
 *  - For backward pass, we need to multiply out everything except the item
 *    whose gradient we are computing (like in typical "product rule of logs").
 *    We'll store partial products or use logs for stability if desired.
 */
template<typename T>
class ProdNode : public Node {
public:
    ProdNode(const Tensor<T>& input, int dim, Tensor<T>& output)
        : input_impl_(input.impl_)
        , output_impl_(output.impl_)
        , input_shape_(input.shape())
        , dim_(dim)
    {
        output_shape_ = output.shape();

        const auto& in_data = input_impl_->data();
        auto& out_data = output_impl_->data();

        if (dim_ == -1) {
            // Single scalar: global product
            T product = 1;
            for (auto val : in_data) {
                product *= val;
            }
            out_data.resize(1);
            out_data[0] = product;
        } else {
            size_t out_size = 1;
            for (auto s : output_shape_) out_size *= s;
            out_data.resize(out_size);

            auto in_strides = computeStrides(input_shape_);
            auto out_strides = computeStrides(output_shape_);

            for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
                auto coords = unravelIndex(out_idx, output_shape_);

                T product = 1;
                for (size_t pos = 0; pos < input_shape_[dim_]; ++pos) {
                    coords[dim_] = pos;
                    size_t in_idx = ravelIndex(coords, in_strides);
                    product *= in_data[in_idx];
                }
                out_data[out_idx] = product;
            }
        }
    }

    std::string node_type() const override { return "Prod"; }

    void backward() override {
        if (!input_impl_->requires_grad()) return;

        auto& in_grad = input_impl_->grad();
        const auto& in_data = input_impl_->data();
        const auto& out_data = output_impl_->data();
        const auto& out_grad = output_impl_->grad();

        // For simplicity, do a partial re-run of the forward logic
        // to get the product excluding the current element. 
        // This is not the most efficient approach (we might store partial products).
        // But it's straightforward.

        if (dim_ == -1) {
            // single scalar
            T product = out_data[0]; // The entire product
            for (size_t i = 0; i < in_data.size(); ++i) {
                if (in_data[i] != 0) {
                    // d/dx [product of all] = product / x
                    in_grad[i] += out_grad[0] * (product / in_data[i]);
                }
                else {
                    // If in_data[i] == 0, you'd need a separate logic 
                    // (the derivative is the product of all others).
                    // We'll do a naive approach: re-multiply all except the current.
                    T partial = 1;
                    for (size_t j = 0; j < in_data.size(); ++j) {
                        if (j != i) {
                            partial *= in_data[j];
                        }
                    }
                    in_grad[i] += out_grad[0] * partial;
                }
            }
        } else {
            // partial dimension
            auto in_strides = computeStrides(input_shape_);
            auto out_strides = computeStrides(output_shape_);
            size_t in_size = in_data.size();

            // For each input element, we see which slice it belongs to in the output
            // Then re-multiply except that element.
            for (size_t i = 0; i < in_size; ++i) {
                auto coords = unravelIndex(i, input_shape_);
                // find the out_idx
                size_t old_val = coords[dim_];
                coords[dim_] = 0;
                size_t out_idx = ravelIndex(coords, out_strides);

                T product_slice = out_data[out_idx]; // product for that entire slice
                T gradient_contribution = 0;

                if (in_data[i] != 0) {
                    gradient_contribution = out_grad[out_idx] * (product_slice / in_data[i]);
                } else {
                    // re-compute product excluding current index
                    T partial = 1;
                    for (size_t pos = 0; pos < input_shape_[dim_]; ++pos) {
                        if (pos == old_val) continue; // exclude the current
                        coords[dim_] = pos;
                        size_t other_in_idx = ravelIndex(coords, in_strides);
                        partial *= in_data[other_in_idx];
                    }
                    gradient_contribution = out_grad[out_idx] * partial;
                }

                in_grad[i] += gradient_contribution;
            }
        }
    }

private:
    std::shared_ptr<detail::TensorImpl<T>> input_impl_;
    std::shared_ptr<detail::TensorImpl<T>> output_impl_;
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape_;
    int dim_;
};

///////////////////////////////////////////////////////////
// (3) Free functions for creating reduction operations
///////////////////////////////////////////////////////////

/**
 * sum(input, dim = -1)
 * If dim = -1, sum all dims => scalar.
 * Otherwise, sum over a particular dim => that dim in the output shape is 1.
 */
template<typename T>
Tensor<T> sum(const Tensor<T>& input, int dim = -1) {
    // Determine output shape
    std::vector<size_t> out_shape = input.shape();
    if (dim == -1) {
        out_shape = {1}; // reduce all dims
    } else {
        out_shape[dim] = 1; // reduce just one dim
    }

    // Create output
    Tensor<T> output(out_shape);

    // Autograd
    if (input.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<SumNode<T>>(input, dim, output);
        ComputationGraph::getInstance().add_node(node);
    }

    // Forward pass: stable summation
    const auto& in_data = input.data();
    auto& out_data = output.data();
    out_data.clear();

    if (dim == -1) {
        // Single scalar
        out_data.resize(1);
        out_data[0] = stable_accumulate(in_data);
    } else {
        // Partial sum across dimension `dim`
        size_t out_size = 1;
        for (auto s : out_shape) out_size *= s;
        out_data.resize(out_size);

        auto in_shape = input.shape();
        auto in_strides = computeStrides(in_shape);
        auto out_strides = computeStrides(out_shape);

        for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
            auto coords = unravelIndex(out_idx, out_shape);
            T sum_val = 0;
            T c = 0; // Kahan compensation
            for (size_t pos = 0; pos < in_shape[dim]; ++pos) {
                coords[dim] = pos;
                size_t in_idx = ravelIndex(coords, in_strides);
                T y = in_data[in_idx] - c;
                T t = sum_val + y;
                c = (t - sum_val) - y;
                sum_val = t;
            }
            out_data[out_idx] = sum_val;
        }
    }
    return output;
}

/**
 * mean(input, dim = -1)
 */
template<typename T>
Tensor<T> mean(const Tensor<T>& input, int dim = -1) {
    // output shape logic is the same as sum
    std::vector<size_t> out_shape = input.shape();
    if (dim == -1) {
        out_shape = {1};
    } else {
        out_shape[dim] = 1;
    }

    Tensor<T> output(out_shape);

    // Autograd
    if (input.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<MeanNode<T>>(input, dim, output);
        ComputationGraph::getInstance().add_node(node);
    }

    // Forward pass
    // 1) Summation
    auto sum_tensor = sum(input, dim);
    // 2) Divide by number of elements along that dimension
    size_t count = 1;
    if (dim == -1) {
        // all dims
        for (auto s : input.shape()) count *= s;
    } else {
        count = input.shape()[dim];
    }

    auto& sum_data = sum_tensor.data();
    auto& out_data = output.data();
    out_data.resize(sum_data.size());

    for (size_t i = 0; i < sum_data.size(); ++i) {
        out_data[i] = sum_data[i] / static_cast<T>(count);
    }

    return output;
}

/**
 * max(input, dim = -1)
 */
template<typename T>
Tensor<T> max(const Tensor<T>& input, int dim = -1) {
    // output shape
    std::vector<size_t> out_shape = input.shape();
    if (dim == -1) {
        out_shape = {1};
    } else {
        out_shape[dim] = 1;
    }

    Tensor<T> output(out_shape);

    // Autograd
    if (input.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<MaxNode<T>>(input, dim, output);
        ComputationGraph::getInstance().add_node(node);
    } else {
        // If no grad is needed, we must still do the forward pass here
        // (since in that case the node won't exist).
        // Alternatively, you can keep the forward pass in the Node constructor.
        // But let's do it the same as the Node for consistency.
        const auto& in_data = input.data();
        auto& out_data = output.data();
        if (dim == -1) {
            // global max
            size_t max_idx = 0;
            T max_val = in_data[0];
            for (size_t i = 1; i < in_data.size(); ++i) {
                if (in_data[i] > max_val) {
                    max_val = in_data[i];
                    max_idx = i;
                }
            }
            out_data.resize(1);
            out_data[0] = max_val;
        } else {
            size_t out_size = 1;
            for (auto s : out_shape) out_size *= s;
            out_data.resize(out_size);

            auto in_shape = input.shape();
            auto in_strides = computeStrides(in_shape);
            auto out_strides = computeStrides(out_shape);

            for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
                auto coords = unravelIndex(out_idx, out_shape);

                T max_val = std::numeric_limits<T>::lowest();

                for (size_t pos = 0; pos < in_shape[dim]; ++pos) {
                    coords[dim] = pos;
                    size_t in_idx = ravelIndex(coords, in_strides);
                    if (in_data[in_idx] > max_val) {
                        max_val = in_data[in_idx];
                    }
                }
                out_data[out_idx] = max_val;
            }
        }
    }

    return output;
}

/**
 * min(input, dim = -1)
 */
template<typename T>
Tensor<T> min(const Tensor<T>& input, int dim = -1) {
    std::vector<size_t> out_shape = input.shape();
    if (dim == -1) {
        out_shape = {1};
    } else {
        out_shape[dim] = 1;
    }

    Tensor<T> output(out_shape);

    // Autograd
    if (input.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<MinNode<T>>(input, dim, output);
        ComputationGraph::getInstance().add_node(node);
    } else {
        // Forward pass if no grad
        const auto& in_data = input.data();
        auto& out_data = output.data();
        if (dim == -1) {
            // global min
            size_t min_idx = 0;
            T min_val = in_data[0];
            for (size_t i = 1; i < in_data.size(); ++i) {
                if (in_data[i] < min_val) {
                    min_val = in_data[i];
                    min_idx = i;
                }
            }
            out_data.resize(1);
            out_data[0] = min_val;
        } else {
            size_t out_size = 1;
            for (auto s : out_shape) out_size *= s;
            out_data.resize(out_size);

            auto in_shape = input.shape();
            auto in_strides = computeStrides(in_shape);
            auto out_strides = computeStrides(out_shape);

            for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
                auto coords = unravelIndex(out_idx, out_shape);

                T min_val = std::numeric_limits<T>::max();
                for (size_t pos = 0; pos < in_shape[dim]; ++pos) {
                    coords[dim] = pos;
                    size_t in_idx = ravelIndex(coords, in_strides);
                    if (in_data[in_idx] < min_val) {
                        min_val = in_data[in_idx];
                    }
                }
                out_data[out_idx] = min_val;
            }
        }
    }

    return output;
}

/**
 * prod(input, dim = -1)
 */
template<typename T>
Tensor<T> prod(const Tensor<T>& input, int dim = -1) {
    std::vector<size_t> out_shape = input.shape();
    if (dim == -1) {
        out_shape = {1};
    } else {
        out_shape[dim] = 1;
    }

    Tensor<T> output(out_shape);

    if (input.requires_grad()) {
        output.set_requires_grad(true);
        auto node = std::make_shared<ProdNode<T>>(input, dim, output);
        ComputationGraph::getInstance().add_node(node);
    } else {
        // Forward pass if no grad
        const auto& in_data = input.data();
        auto& out_data = output.data();
        if (dim == -1) {
            // global product
            T product = 1;
            for (auto val : in_data) {
                product *= val;
            }
            out_data.resize(1);
            out_data[0] = product;
        } else {
            size_t out_size = 1;
            for (auto s : out_shape) out_size *= s;
            out_data.resize(out_size);

            auto in_shape = input.shape();
            auto in_strides = computeStrides(in_shape);
            auto out_strides = computeStrides(out_shape);

            for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
                auto coords = unravelIndex(out_idx, out_shape);

                T product = 1;
                for (size_t pos = 0; pos < in_shape[dim]; ++pos) {
                    coords[dim] = pos;
                    size_t in_idx = ravelIndex(coords, in_strides);
                    product *= in_data[in_idx];
                }
                out_data[out_idx] = product;
            }
        }
    }

    return output;
}

} // namespace ops
} // namespace dl
