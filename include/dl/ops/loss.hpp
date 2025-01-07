#pragma once

#include "../tensor.hpp"
#include "../autograd.hpp"
#include <cmath>
#include <memory>

namespace dl {
namespace ops {

template<typename T>
class MSELossNode : public Node {
public:
    MSELossNode(const Tensor<T>& pred, const Tensor<T>& target, Tensor<T>& loss)
        : pred_(pred), target_(target), loss_(loss) {}

    void backward() override {
        if (pred_.requires_grad()) {
            auto& pred_grad = const_cast<Tensor<T>&>(pred_).grad();
            const auto& pred_data = pred_.data();
            const auto& target_data = target_.data();
            const auto& loss_grad = loss_.grad();
            
            T scale = T(2) / pred_data.size();
            for (size_t i = 0; i < pred_data.size(); ++i) {
                pred_grad[i] += scale * (pred_data[i] - target_data[i]) * loss_grad[0];
            }
        }
    }

private:
    Tensor<T> pred_;
    Tensor<T> target_;
    Tensor<T> loss_;
};

template<typename T>
class CrossEntropyLossNode : public Node {
public:
    CrossEntropyLossNode(const Tensor<T>& pred, const Tensor<T>& target, Tensor<T>& result)
        : pred_(pred), target_(target), result_(result) {}

    void backward() override {
        if (pred_.requires_grad()) {
            auto& pred_grad = pred_.grad();
            const auto& pred_data = pred_.data();
            const auto& target_data = target_.data();
            const auto& result_grad = result_.grad();
            
            T scale = -T(1) / pred_data.size();
            for (size_t i = 0; i < pred_grad.size(); ++i) {
                pred_grad[i] += scale * (target_data[i] / pred_data[i]) * result_grad[0];
            }
        }
    }

private:
    const Tensor<T>& pred_;
    const Tensor<T>& target_;
    const Tensor<T>& result_;
};

template<typename T>
Tensor<T> mse_loss(const Tensor<T>& pred, const Tensor<T>& target) {
    const auto& pred_data = pred.data();
    const auto& target_data = target.data();
    
    if (pred_data.size() != target_data.size()) {
        throw std::runtime_error("Prediction and target tensors must have the same size");
    }
    
    T loss = T(0);
    for (size_t i = 0; i < pred_data.size(); ++i) {
        T diff = pred_data[i] - target_data[i];
        loss += diff * diff;
    }
    loss /= pred_data.size();
    
    Tensor<T> loss_tensor({1});
    loss_tensor.data()[0] = loss;
    
    // Set requires_grad if prediction requires grad
    if (pred.requires_grad()) {
        loss_tensor.set_requires_grad(true);
        auto node = std::make_shared<MSELossNode<T>>(pred, target, loss_tensor);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return loss_tensor;
}

} // namespace ops

// Loss function implementations
template<typename T>
Tensor<T> cross_entropy_loss(const Tensor<T>& pred, const Tensor<T>& target) {
    if (pred.shape() != target.shape()) {
        throw std::runtime_error("Prediction and target shapes must match");
    }
    
    Tensor<T> result({1}); // Scalar output
    auto& result_data = result.data();
    const auto& pred_data = pred.data();
    const auto& target_data = target.data();
    
    T sum = T(0);
    for (size_t i = 0; i < pred_data.size(); ++i) {
        sum -= target_data[i] * std::log(pred_data[i] + T(1e-7));
    }
    result_data[0] = sum / pred_data.size();
    
    auto node = std::make_shared<ops::CrossEntropyLossNode<T>>(pred, target, result);
    ComputationGraph::getInstance().addNode(node);
    
    return result;
}

} // namespace dl
