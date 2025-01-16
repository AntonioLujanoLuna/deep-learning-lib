#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include "../utils/utils.hpp"
#include "matrix_ops.hpp"  // if needed
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>  // for std::max, std::min
#include <iostream>

namespace dl {
namespace ops {

//
// 1) MSELossNode
//
template<typename T>
class MSELossNode : public Node{
public:
    // Constructor: link in DAG and do not do the forward pass (the free function does it).
    MSELossNode(const Tensor<T>& predicted, const Tensor<T>& target,
                const std::shared_ptr<Tensor<T>>& output)
        : predicted_(predicted)
        , target_(target)
        , output_(output) 
    {
        // 1) The output is produced by *this* node
        output_->setGradFn(this->shared_from_this());

        // 2) If the predicted Tensor has a gradFn, link as parent
        if (auto parent = predicted->gradFn().lock()) {
            parents_.push_back(parent);
            parent->children_.push_back(this->shared_from_this());
        }

        // 3) Register node with the ComputationGraph
        ComputationGraph::getInstance().addNode(this->shared_from_this());
    }

    std::string node_type() const override {
        return "MSELossNode";
    }

    void backward() override {
        // If predicted does not require grad, do nothing
        if (!predicted_.requires_grad()) {
            return;
        }

        auto& pred_grad = const_cast<Tensor<T>&>(predicted_).grad();
        const auto& pred_data   = predicted_.data();
        const auto& target_data = target_.data();
        const auto& out_grad    = output_->grad();

        // Typically out_grad[0] is 1, but can be user-modified
        T scale = T(2.0) * out_grad[0] / pred_data.size();
        for (size_t i = 0; i < pred_data.size(); ++i) {
            pred_grad[i] += scale * (pred_data[i] - target_data[i]);
        }
    }

private:
    const Tensor<T>& predicted_;
    const Tensor<T>& target_;
    std::shared_ptr<Tensor<T>> output_;
};

//
// 2) BCELossNode
//
template<typename T>
class BCELossNode : public Node {
public:
    BCELossNode(const std::shared_ptr<Tensor<T>>& predicted, 
                const Tensor<T>& target, 
                const std::shared_ptr<Tensor<T>>& output)
        : predicted_(predicted)
        , target_(target)
        , output_(output)
    {
        // 1) The output is produced by *this* node
        output_->setGradFn(this->shared_from_this());

        // 2) If predicted has a gradFn, link it
        if (auto parent = predicted->gradFn().lock()) {
            parents_.push_back(parent);
            parent->children_.push_back(this->shared_from_this());
        }

        // 3) Register node
        ComputationGraph::getInstance().addNode(this->shared_from_this());
    }

    std::string node_type() const override {
        return "BCELossNode";
    }

    void backward() override {
        // If predicted does not require grad, do nothing
        if (!predicted_->requires_grad()) {
            return;
        }

        auto& pred_grad    = predicted_->grad();
        const auto& pred_data   = predicted_->data();
        const auto& target_data = target_.data();
        const auto& out_grad    = output_->grad();

        T scale = out_grad[0] / pred_data.size();
        for (size_t i = 0; i < pred_data.size(); ++i) {
            T p = std::max(std::min(pred_data[i], T(1) - T(1e-7)), T(1e-7));
            pred_grad[i] += scale * (p - target_data[i]) / (p * (1 - p));
        }
    }

private:
    std::shared_ptr<Tensor<T>> predicted_;
    const Tensor<T>& target_;
    std::shared_ptr<Tensor<T>> output_;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/**
 * mse_loss - Creates a scalar that is the MSE between predicted and target.
 */
template<typename T>
std::shared_ptr<Tensor<T>> mse_loss(const Tensor<T>& predicted, const Tensor<T>& target) {
    // 1) Check shapes
    if (predicted.shape() != target.shape()) {
        throw std::runtime_error("Predicted and target shapes must match for MSE loss");
    }
    
    // 2) Create output (a scalar)
    auto output = std::make_shared<Tensor<T>>(std::vector<size_t>{1});
    // We'll do the forward pass here
    output->set_requires_grad(true);

    // 3) Forward pass
    T sum_sq_error = T(0);
    const auto& pred_data   = predicted.data();
    const auto& target_data = target.data();
    for (size_t i = 0; i < pred_data.size(); ++i) {
        T diff = pred_data[i] - target_data[i];
        sum_sq_error += diff * diff;
    }
    // MSE
    output->data()[0] = sum_sq_error / static_cast<T>(pred_data.size());

    // 4) If predicted requires grad, create an MSELossNode
    if (predicted.requires_grad()) {
        auto node = std::make_shared<MSELossNode<T>>(predicted, target, output);
    }

    return output;
}

// Overload if user passes predicted as a shared_ptr
template<typename T>
std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& predicted, const Tensor<T>& target) {
    return mse_loss(*predicted, target);
}

/**
 * binary_cross_entropy - Creates a scalar that is the BCE between predicted and target.
 * predicted is typically in the range (0,1). target is 0 or 1 usually.
 */
template<typename T>
std::shared_ptr<Tensor<T>> binary_cross_entropy(std::shared_ptr<Tensor<T>> predicted,
                                                const Tensor<T>& target) 
{
    // 1) Check shapes
    if (predicted->shape() != target.shape()) {
        throw std::runtime_error("Predicted and target shapes must match for BCE loss");
    }

    // 2) Create output (a scalar)
    auto output = std::make_shared<Tensor<T>>(std::vector<size_t>{1});
    output->set_requires_grad(true);

    // 3) Forward pass
    T total_loss = T(0);
    const auto& pred_data   = predicted->data();
    const auto& target_data = target.data();

    for (size_t i = 0; i < pred_data.size(); ++i) {
        // clamp p in [1e-7, 1 - 1e-7]
        T p = std::max(std::min(pred_data[i], T(1) - T(1e-7)), T(1e-7));
        total_loss += -(target_data[i] * std::log(p) + (T(1) - target_data[i]) * std::log(T(1) - p));
    }
    output->data()[0] = total_loss / static_cast<T>(pred_data.size());

    // 4) If predicted requires grad, create BCELossNode
    if (predicted->requires_grad()) {
        auto node = std::make_shared<BCELossNode<T>>(predicted, target, output);
    }

    return output;
}

} // namespace ops
} // namespace dl
