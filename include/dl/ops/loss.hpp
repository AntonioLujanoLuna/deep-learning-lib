#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "../tensor.hpp"
#include "../autograd.hpp"
#include "matrix_ops.hpp"

namespace dl {
namespace ops {

template<typename T>
std::shared_ptr<Tensor<T>> mse_loss(const Tensor<T>& predicted, const Tensor<T>& target) {
    // Check shapes match
    if (predicted.shape() != target.shape()) {
        throw std::runtime_error("Predicted and target shapes must match for MSE loss");
    }
    
    // Create output tensor
    std::vector<size_t> output_shape{1};
    auto output = std::make_shared<Tensor<T>>(output_shape);
    output->set_requires_grad(true);
    
    // Calculate MSE loss
    T sum_squared_error = 0;
    const auto& pred_data = predicted.data();
    const auto& target_data = target.data();
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        T diff = pred_data[i] - target_data[i];
        sum_squared_error += diff * diff;
    }
    
    output->data()[0] = sum_squared_error / static_cast<T>(pred_data.size());
    
    // Create and register backward node if needed
    if (predicted.requires_grad()) {
        struct MSELossNode : public Node {
            MSELossNode(const Tensor<T>& pred, const Tensor<T>& tgt, std::shared_ptr<Tensor<T>> out)
                : predicted(pred), target(tgt), output(out) {}
            
            void backward() override {
                if (!predicted.requires_grad()) return;
                
                auto& pred_grad = const_cast<Tensor<T>&>(predicted).grad();
                const auto& pred_data = predicted.data();
                const auto& target_data = target.data();
                const auto& out_grad = output->grad();
                
                T scale = 2.0 * out_grad[0] / pred_data.size();
                for (size_t i = 0; i < pred_data.size(); ++i) {
                    pred_grad[i] += scale * (pred_data[i] - target_data[i]);
                }
            }
            
            std::string node_type() const override {
                return "MSELossNode";
            }
            
            const Tensor<T>& predicted;
            const Tensor<T>& target;
            std::shared_ptr<Tensor<T>> output;
        };
        
        ComputationGraph::getInstance().addNode(std::make_shared<MSELossNode>(predicted, target, output));
    }
    
    return output;
}

// Overload for shared_ptr input
template<typename T>
std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& predicted, const Tensor<T>& target) {
    return mse_loss(*predicted, target);
}

template<typename T>
std::shared_ptr<Tensor<T>> binary_cross_entropy(std::shared_ptr<Tensor<T>> predicted, const Tensor<T>& target) {
    // Check shapes match
    if (predicted->shape() != target.shape()) {
        throw std::runtime_error("Predicted and target shapes must match for BCE loss");
    }
    
    // Create output tensor
    std::vector<size_t> output_shape{1};
    auto output = std::make_shared<Tensor<T>>(output_shape);
    output->set_requires_grad(true);
    
    // Calculate BCE loss
    T total_loss = 0;
    const auto& pred_data = predicted->data();
    const auto& target_data = target.data();
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        T p = std::max(std::min(pred_data[i], T(1) - T(1e-7)), T(1e-7));
        total_loss += -(target_data[i] * std::log(p) + (1 - target_data[i]) * std::log(1 - p));
    }
    
    output->data()[0] = total_loss / static_cast<T>(pred_data.size());
    
    // Create and register backward node if needed
    if (predicted->requires_grad()) {
        struct BCELossNode : public Node {
            BCELossNode(std::shared_ptr<Tensor<T>> pred, const Tensor<T>& tgt, std::shared_ptr<Tensor<T>> out)
                : predicted(pred), target(tgt), output(out) {}
            
            void backward() override {
                auto& pred_grad = predicted->grad();
                const auto& pred_data = predicted->data();
                const auto& target_data = target.data();
                const auto& out_grad = output->grad();
                
                T scale = out_grad[0] / pred_data.size();
                for (size_t i = 0; i < pred_data.size(); ++i) {
                    T p = std::max(std::min(pred_data[i], T(1) - T(1e-7)), T(1e-7));
                    pred_grad[i] += scale * (p - target_data[i]) / (p * (1 - p));
                }
            }
            
            std::string node_type() const override {
                return "BCELossNode";
            }
            
            std::shared_ptr<Tensor<T>> predicted;
            const Tensor<T>& target;
            std::shared_ptr<Tensor<T>> output;
        };
        
        ComputationGraph::getInstance().addNode(std::make_shared<BCELossNode>(predicted, target, output));
    }
    
    return output;
}

} // namespace ops
} // namespace dl