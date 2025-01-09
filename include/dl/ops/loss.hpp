#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <sstream>
#include <iomanip>
#include "../tensor.hpp"
#include "../autograd.hpp"
#include "../utils.hpp"

namespace dl {
namespace ops {

template<typename T>
class BCELossNode : public Node {
public:
    BCELossNode(Tensor<T>& predicted, Tensor<T>& target, Tensor<T>& output)
        : predicted_tensor_(predicted)
        , target_tensor_(target)
        , output_tensor_(output)
        , predicted_shape_(predicted.shape())
        , target_shape_(target.shape())
        , output_shape_(output.shape())
        , predicted_size_(std::accumulate(predicted_shape_.begin(), predicted_shape_.end(), size_t(1), std::multiplies<size_t>()))
        , target_size_(std::accumulate(target_shape_.begin(), target_shape_.end(), size_t(1), std::multiplies<size_t>()))
        , output_size_(std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies<size_t>())) {
        
        if (predicted_tensor_.shape().empty() || target_tensor_.shape().empty() || 
            output_tensor_.shape().empty() || output_tensor_.shape() != std::vector<size_t>{1, 1}) {
            throw std::runtime_error("Invalid tensor shapes in BCELossNode. Predicted shape: " + 
                utils::shape_to_string(predicted_tensor_.shape()) + ", Target shape: " + 
                utils::shape_to_string(target_tensor_.shape()) + ", Output shape: " + 
                utils::shape_to_string(output_tensor_.shape()));
        }
    }

    std::string node_type() const override {
        return "BCELoss";
    }

    void backward() override {
        if (predicted_tensor_.requires_grad()) {
            auto& predicted_grad = predicted_tensor_.grad();
            const auto& predicted_data = predicted_tensor_.data();
            const auto& target_data = target_tensor_.data();
            const auto& output_grad = output_tensor_.grad();
            
            // Initialize output gradient if empty
            if (output_grad.empty()) {
                output_tensor_.grad().assign(1, T(1));  
            }
            
            // Initialize predicted gradient if needed
            if (predicted_grad.size() != predicted_size_) {
                predicted_grad.resize(predicted_size_, T(0));
            }
            
            // Clear gradients before computing them
            std::fill(predicted_grad.begin(), predicted_grad.end(), T(0));
            
            // Compute gradient: -(t/p - (1-t)/(1-p))
            const T output_grad_val = output_tensor_.grad()[0];
            const T eps = T(1e-7);  // Small epsilon for numerical stability
            
            for (size_t i = 0; i < predicted_data.size(); ++i) {
                T p = std::max(std::min(predicted_data[i], T(1) - eps), eps);
                T t = target_data[i];
                
                // Compute gradient for binary cross entropy
                T grad_i = -(t/p - (T(1)-t)/(T(1)-p));
                predicted_grad[i] = grad_i * output_grad_val;  
            }
        }
    }

private:
    // Store references to avoid copying
    Tensor<T>& predicted_tensor_;
    Tensor<T>& target_tensor_;
    Tensor<T>& output_tensor_;
    std::vector<size_t> predicted_shape_;
    std::vector<size_t> target_shape_;
    std::vector<size_t> output_shape_;
    size_t predicted_size_;
    size_t target_size_;
    size_t output_size_;
};

template<typename T>
class MSELossNode : public Node {
public:
    MSELossNode(Tensor<T>& predicted, Tensor<T>& target, Tensor<T>& output)
        : predicted_tensor_(predicted), target_tensor_(target), output_tensor_(output) {
        if (predicted_tensor_.shape().empty() || target_tensor_.shape().empty() || 
            output_tensor_.shape().empty() || output_tensor_.shape() != std::vector<size_t>{1, 1}) {
            throw std::runtime_error("Invalid tensor shapes in MSELossNode. Predicted shape: " + 
                utils::shape_to_string(predicted_tensor_.shape()) + ", Target shape: " + 
                utils::shape_to_string(target_tensor_.shape()) + ", Loss shape: " + 
                utils::shape_to_string(output_tensor_.shape()));
        }
        
        // Store shapes for later use
        pred_shape_ = predicted_tensor_.shape();
        target_shape_ = target_tensor_.shape();
        loss_shape_ = output_tensor_.shape();
        
        // Store initial sizes
        pred_size_ = std::accumulate(pred_shape_.begin(), pred_shape_.end(), size_t(1), std::multiplies<size_t>());
        target_size_ = std::accumulate(target_shape_.begin(), target_shape_.end(), size_t(1), std::multiplies<size_t>());
        loss_size_ = std::accumulate(loss_shape_.begin(), loss_shape_.end(), size_t(1), std::multiplies<size_t>());
    }

    std::string node_type() const override {
        return "MSELoss";
    }

    void backward() override {
        if (predicted_tensor_.requires_grad()) {
            auto& predicted_grad = const_cast<Tensor<T>&>(predicted_tensor_).grad();
            const auto& predicted_data = predicted_tensor_.data();
            const auto& target_data = target_tensor_.data();
            auto& loss_grad = output_tensor_.grad();
            
            // Initialize loss gradient to 1 if empty
            if (loss_grad.empty()) {
                loss_grad.assign(1, T(1));  
            }
            
            // Print debug info for first few iterations
            static int backward_count = 0;
            bool print_debug = backward_count < 5;
            
            if (print_debug) {
                std::cout << "\n=== MSE Loss Backward Pass " << backward_count << " ===" << std::endl;
                std::cout << "Predicted shape: " << utils::shape_to_string(predicted_tensor_.shape()) << std::endl;
                std::cout << "Target shape: " << utils::shape_to_string(target_tensor_.shape()) << std::endl;
                std::cout << "Loss grad value (after init): " << loss_grad[0] << std::endl;
                
                // Print first few values
                std::cout << "First 3 values:" << std::endl;
                for (size_t i = 0; i < std::min(size_t(3), predicted_data.size()); ++i) {
                    std::cout << "  pred[" << i << "] = " << predicted_data[i] 
                             << ", target[" << i << "] = " << target_data[i] 
                             << ", diff = " << (predicted_data[i] - target_data[i]) << std::endl;
                }
            }
            
            // Initialize predicted gradient if needed
            if (predicted_grad.size() != pred_size_) {
                predicted_grad.resize(pred_size_, T(0));
            }
            
            // Clear gradients before computing them
            std::fill(predicted_grad.begin(), predicted_grad.end(), T(0));
            
            // Compute gradients with proper scaling
            const T scale = T(2);  // 2 * (pred - target) for MSE derivative
            for (size_t i = 0; i < predicted_data.size(); ++i) {
                T diff = predicted_data[i] - target_data[i];
                T grad_i = scale * diff * loss_grad[0];
                predicted_grad[i] = grad_i;  
                
                if (print_debug && i < 3) {
                    std::cout << "  grad[" << i << "] = " << grad_i 
                             << " (scale=" << scale 
                             << ", diff=" << diff
                             << ", loss_grad=" << loss_grad[0] << ")" << std::endl;
                }
            }
            
            backward_count++;
            
            if (print_debug) {
                std::cout << "First few predicted grads after MSE backward: ";
                for (size_t i = 0; i < std::min(size_t(3), predicted_grad.size()); ++i) {
                    std::cout << predicted_grad[i] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

private:
    // Store by value to avoid dangling references
    Tensor<T> predicted_tensor_;
    Tensor<T> target_tensor_;
    Tensor<T>& output_tensor_;  // Keep reference since this is managed by computation graph
    std::vector<size_t> pred_shape_;
    std::vector<size_t> target_shape_;
    std::vector<size_t> loss_shape_;
    size_t pred_size_;
    size_t target_size_;
    size_t loss_size_;
};

template<typename T>
Tensor<T> mse_loss(const Tensor<T>& predicted, const Tensor<T>& target) {
    // Create output tensor and store it in the graph
    auto output = std::make_shared<Tensor<T>>(std::vector<size_t>{1, 1});
    output->set_requires_grad(true);
    
    const auto& p = predicted.data();
    const auto& t = target.data();
    auto& out = output->data();
    out.resize(1);  
    
    T total_loss = T(0);
    for (size_t i = 0; i < p.size(); ++i) {
        T diff = p[i] - t[i];
        total_loss += diff * diff;
    }
    out[0] = total_loss / static_cast<T>(p.size());
    
    // Initialize gradient vector but don't set value yet
    output->grad().resize(1);
    
    if (predicted.requires_grad()) {
        // Create a copy of the predicted tensor to avoid const_cast
        auto stored_predicted = std::make_shared<Tensor<T>>(predicted);
        auto stored_target = std::make_shared<Tensor<T>>(target);
        
        // Store tensors in computation graph
        auto& graph = ComputationGraph::getInstance();
        graph.storeTensorPtr(stored_predicted);
        graph.storeTensorPtr(stored_target);
        graph.storeTensorPtr(output);
        
        auto node = std::make_shared<MSELossNode<T>>(*stored_predicted, *stored_target, *output);
        graph.addNode(node);
    }
    
    return *output;
}

template<typename T>
Tensor<T> binary_cross_entropy(const Tensor<T>& predicted, const Tensor<T>& target) {
    // Create output tensor and store it in the graph
    auto output = std::make_shared<Tensor<T>>(std::vector<size_t>{1, 1});
    output->set_requires_grad(true);
    
    const auto& p = predicted.data();
    const auto& t = target.data();
    auto& out = output->data();
    out.resize(1);
    
    T total_loss = T(0);
    const T eps = T(1e-7);  // Small epsilon for numerical stability
    
    for (size_t i = 0; i < p.size(); ++i) {
        T p_i = std::max(std::min(p[i], T(1) - eps), eps);
        T t_i = t[i];
        total_loss += -t_i * std::log(p_i) - (T(1) - t_i) * std::log(T(1) - p_i);
    }
    out[0] = total_loss / static_cast<T>(p.size());
    
    // Initialize gradient vector but don't set value yet
    output->grad().resize(1);
    
    if (predicted.requires_grad()) {
        // Create a copy of the predicted tensor to avoid const_cast
        auto stored_predicted = std::make_shared<Tensor<T>>(predicted);
        auto stored_target = std::make_shared<Tensor<T>>(target);
        
        // Store tensors in computation graph
        auto& graph = ComputationGraph::getInstance();
        graph.storeTensorPtr(stored_predicted);
        graph.storeTensorPtr(stored_target);
        graph.storeTensorPtr(output);
        
        auto node = std::make_shared<BCELossNode<T>>(*stored_predicted, *stored_target, *output);
        graph.addNode(node);
    }
    
    return *output;
}

} // namespace ops
} // namespace dl
