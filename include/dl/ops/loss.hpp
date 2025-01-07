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
    BCELossNode(const Tensor<T>& predicted, const Tensor<T>& target, Tensor<T>& output)
        : predicted_(predicted)
        , target_(target)
        , output_(output)
        , predicted_shape_(predicted.shape())
        , target_shape_(target.shape())
        , output_shape_(output.shape())
        , predicted_size_(std::accumulate(predicted_shape_.begin(), predicted_shape_.end(), size_t(1), std::multiplies<size_t>()))
        , target_size_(std::accumulate(target_shape_.begin(), target_shape_.end(), size_t(1), std::multiplies<size_t>()))
        , output_size_(std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies<size_t>()))
        , predicted_tensor_(predicted)
        , target_tensor_(target)
        , output_tensor_(output) {
        
        std::cout << "\nBCELossNode constructor:" << std::endl;
        std::cout << "  Predicted tensor:" << std::endl;
        std::cout << "    Shape: " << utils::shape_to_string(predicted_shape_) << std::endl;
        std::cout << "    Requires grad: " << std::boolalpha << predicted_.requires_grad() << std::endl;
        std::cout << "    Data size: " << predicted_.data().size() << std::endl;
        std::cout << "    Grad size: " << predicted_.grad().size() << std::endl;
        
        std::cout << "  Target tensor:" << std::endl;
        std::cout << "    Shape: " << utils::shape_to_string(target_shape_) << std::endl;
        std::cout << "    Data size: " << target_.data().size() << std::endl;
        
        std::cout << "  Output tensor:" << std::endl;
        std::cout << "    Shape: " << utils::shape_to_string(output_shape_) << std::endl;
        std::cout << "    Requires grad: " << output_.requires_grad() << std::endl;
        std::cout << "    Data size: " << output_.data().size() << std::endl;
        std::cout << "    Grad size: " << output_.grad().size() << std::endl;
        
        // Initialize predicted gradient if needed
        if (predicted_.requires_grad() && predicted_.grad().size() != predicted_size_) {
            std::cout << "  Initializing predicted gradient to size " << predicted_size_ << std::endl;
            auto& predicted_grad = const_cast<Tensor<T>&>(predicted_).grad();
            predicted_grad.resize(predicted_size_, T(0));
        }
        
        if (predicted_.shape().empty() || target_.shape().empty() || output_.shape().empty() || output_.shape() != std::vector<size_t>{1, 1}) {
            throw std::runtime_error("Invalid tensor shapes in BCELossNode. Predicted shape: " + utils::shape_to_string(predicted_.shape()) + ", Target shape: " + utils::shape_to_string(target_.shape()) + ", Output shape: " + utils::shape_to_string(output_.shape()));
        }
    }

    std::string node_type() const override {
        return "BCELoss";
    }

    void backward() override {
        std::cout << "\n=== Starting BCE Loss backward ===" << std::endl;
        std::cout << "Debug point 1: Getting shapes" << std::endl;
        std::cout << "Input shape: " << utils::shape_to_string(predicted_shape_) << std::endl;
        std::cout << "Target shape: " << utils::shape_to_string(target_shape_) << std::endl;
        std::cout << "Output shape: " << utils::shape_to_string(output_shape_) << std::endl;
        
        std::cout << "Debug point 2: Getting tensors" << std::endl;
        if (predicted_tensor_.requires_grad()) {
            std::cout << "Predicted tensor requires gradients" << std::endl;
            
            std::cout << "Predicted tensor state:" << std::endl;
            std::cout << "  Shape: " << utils::shape_to_string(predicted_tensor_.shape()) << std::endl;
            std::cout << "  Data size: " << predicted_tensor_.data().size() << std::endl;
            std::cout << "  Requires grad: " << predicted_tensor_.requires_grad() << std::endl;
            
            auto& predicted_grad = predicted_tensor_.grad();
            const auto& predicted_data = predicted_tensor_.data();
            const auto& target_data = target_tensor_.data();
            
            // Get output gradient, initialize if needed
            auto& output_grad = output_tensor_.grad();
            
            std::cout << "Output tensor state:" << std::endl;
            std::cout << "  Shape: " << utils::shape_to_string(output_tensor_.shape()) << std::endl;
            std::cout << "  Data size: " << output_tensor_.data().size() << std::endl;
            std::cout << "  Grad size before init: " << output_grad.size() << std::endl;
            
            if (output_grad.empty() || output_grad.size() != 1) {
                std::cout << "Initializing output gradient to size 1 with value 1" << std::endl;
                output_grad.resize(1, T(1));
            }
            
            // Validate predicted gradient
            std::cout << "Predicted gradient state:" << std::endl;
            std::cout << "  Size before resize: " << predicted_grad.size() << std::endl;
            std::cout << "  Expected size: " << predicted_size_ << std::endl;
            
            if (predicted_grad.size() != predicted_size_) {
                std::cout << "Resizing predicted gradient from " << predicted_grad.size() << " to " << predicted_size_ << std::endl;
                predicted_grad.resize(predicted_size_, T(0));
            }
            
            std::cout << "Debug point 3: Checking sizes" << std::endl;
            std::cout << "Predicted grad size: " << predicted_grad.size() << std::endl;
            std::cout << "Predicted data size: " << predicted_data.size() << std::endl;
            std::cout << "Target data size: " << target_data.size() << std::endl;
            std::cout << "Output grad size: " << output_grad.size() << std::endl;
            
            // Compute gradient: -(t/p - (1-t)/(1-p)) / N
            std::cout << "Debug point 4: Computing gradients" << std::endl;
            const T N = static_cast<T>(predicted_data.size());  // Normalize by batch size
            const T output_grad_val = output_grad[0];  // Cache this value
            
            std::cout << "N = " << N << ", output_grad_val = " << output_grad_val << std::endl;
            
            for (size_t i = 0; i < predicted_data.size(); ++i) {
                // Clip predicted values to avoid numerical instability
                T p = std::max(std::min(predicted_data[i], T(1) - T(1e-7)), T(1e-7));
                T t = target_data[i];
                
                // Compute gradient for binary cross entropy:
                // d(BCE)/dp = -(t/p - (1-t)/(1-p))
                T grad_i = -(t/p - (T(1)-t)/(T(1)-p)) / N;
                
                // Accumulate gradient (use += instead of =)
                predicted_grad[i] += grad_i * output_grad_val;
                
                if (i < 3 || std::isnan(grad_i) || std::isinf(grad_i)) {
                    std::cout << "i=" << i << ": p=" << p << ", t=" << t 
                              << ", grad=" << grad_i 
                              << ", final_grad=" << (grad_i * output_grad_val) << std::endl;
                }
            }
            
            // Validate final gradients
            std::cout << "Debug point 5: Validating final gradients" << std::endl;
            std::cout << "Predicted grad size: " << predicted_grad.size() << std::endl;
            
            for (size_t i = 0; i < predicted_grad.size(); ++i) {
                if (std::isnan(predicted_grad[i]) || std::isinf(predicted_grad[i])) {
                    std::cout << "Warning: Invalid gradient at index " << i 
                              << ": " << predicted_grad[i] << std::endl;
                }
            }
            
            std::cout << "First few gradients:" << std::endl;
            for (size_t i = 0; i < std::min(size_t(3), predicted_grad.size()); ++i) {
                std::cout << "grad[" << i << "] = " << predicted_grad[i] << std::endl;
            }
        } else {
            std::cout << "Predicted tensor does not require gradients" << std::endl;
        }
        
        std::cout << "=== BCE Loss backward completed ===" << std::endl;
    }

private:
    // References to input tensors
    const Tensor<T>& predicted_;
    const Tensor<T>& target_;
    Tensor<T>& output_;
    
    // Store copies to keep tensors alive
    Tensor<T> predicted_tensor_;
    Tensor<T> target_tensor_;
    Tensor<T> output_tensor_;
    
    // Store shapes
    const std::vector<size_t> predicted_shape_;
    const std::vector<size_t> target_shape_;
    const std::vector<size_t> output_shape_;
    
    // Store sizes
    const size_t predicted_size_;
    const size_t target_size_;
    const size_t output_size_;
};

template<typename T>
class MSELossNode : public Node {
public:
    MSELossNode(const Tensor<T>& predicted, const Tensor<T>& target, Tensor<T>& output)
        : predicted_(predicted), target_(target), output_(output) {
        if (predicted_.shape().empty() || target_.shape().empty() || output_.shape().empty() || output_.shape() != std::vector<size_t>{1, 1}) {
            throw std::runtime_error("Invalid tensor shapes in MSELossNode. Predicted shape: " + utils::shape_to_string(predicted_.shape()) + ", Target shape: " + utils::shape_to_string(target_.shape()) + ", Loss shape: " + utils::shape_to_string(output_.shape()));
        }
        
        // Store shapes for later use
        pred_shape_ = predicted_.shape();
        target_shape_ = target_.shape();
        loss_shape_ = output_.shape();
        
        // Store initial sizes
        pred_size_ = std::accumulate(pred_shape_.begin(), pred_shape_.end(), size_t(1), std::multiplies<size_t>());
        target_size_ = std::accumulate(target_shape_.begin(), target_shape_.end(), size_t(1), std::multiplies<size_t>());
        loss_size_ = std::accumulate(loss_shape_.begin(), loss_shape_.end(), size_t(1), std::multiplies<size_t>());
    }

    std::string node_type() const override {
        return "MSELoss";
    }

    void backward() override {
        if (predicted_.requires_grad()) {
            auto& predicted_grad = const_cast<Tensor<T>&>(predicted_).grad();
            const auto& predicted_data = predicted_.data();
            const auto& target_data = target_.data();
            const auto& loss_grad = output_.grad();
            
            // Validate sizes
            if (predicted_grad.size() != pred_size_) {
                throw std::runtime_error("Predicted gradient size mismatch. Expected: " + std::to_string(pred_size_) + ", Got: " + std::to_string(predicted_grad.size()));
            }
            if (predicted_data.size() != pred_size_) {
                throw std::runtime_error("Predicted data size mismatch. Expected: " + std::to_string(pred_size_) + ", Got: " + std::to_string(predicted_data.size()));
            }
            if (target_data.size() != target_size_) {
                throw std::runtime_error("Target data size mismatch. Expected: " + std::to_string(target_size_) + ", Got: " + std::to_string(target_data.size()));
            }
            if (loss_grad.size() != loss_size_) {
                throw std::runtime_error("Loss gradient size mismatch. Expected: " + std::to_string(loss_size_) + ", Got: " + std::to_string(loss_grad.size()));
            }
            
            T scale = T(2) / predicted_data.size();
            for (size_t i = 0; i < predicted_data.size(); ++i) {
                predicted_grad[i] += scale * (predicted_data[i] - target_data[i]) * loss_grad[0];
            }
        }
    }

private:
    const Tensor<T>& predicted_;
    const Tensor<T>& target_;
    Tensor<T>& output_;
    std::vector<size_t> pred_shape_;
    std::vector<size_t> target_shape_;
    std::vector<size_t> loss_shape_;
    size_t pred_size_;
    size_t target_size_;
    size_t loss_size_;
};

template<typename T>
Tensor<T> binary_cross_entropy(const Tensor<T>& predicted, const Tensor<T>& target) {
    std::cout << "\n=== Starting binary_cross_entropy forward ===" << std::endl;
    std::cout << "Predicted requires_grad: " << std::boolalpha << predicted.requires_grad() << std::endl;
    std::cout << "Target requires_grad: " << target.requires_grad() << std::endl;
    std::cout << "Predicted shape: " << utils::shape_to_string(predicted.shape()) << std::endl;
    std::cout << "Target shape: " << utils::shape_to_string(target.shape()) << std::endl;
    
    // Validate input shapes
    if (predicted.shape() != target.shape()) {
        throw std::runtime_error("Shape mismatch: predicted shape " + 
            utils::shape_to_string(predicted.shape()) + " != target shape " + 
            utils::shape_to_string(target.shape()));
    }
    
    // Create output tensor with requires_grad=true
    std::vector<size_t> output_shape{1, 1};
    Tensor<T> output(output_shape);
    output.set_requires_grad(true);
    std::cout << "Created output tensor with shape " << utils::shape_to_string(output.shape()) << std::endl;
    
    // Initialize output gradient to 1 for backward pass
    auto& grad = output.grad();
    std::cout << "Initializing output gradient to size 1 with value 1" << std::endl;
    grad.resize(1, T(1));
    
    std::cout << "Output tensor state:" << std::endl;
    std::cout << "  requires_grad: " << output.requires_grad() << std::endl;
    std::cout << "  gradient size: " << grad.size() << std::endl;
    std::cout << "  gradient value: " << grad[0] << std::endl;
    
    // Create and add node to computation graph if predicted requires gradients
    if (predicted.requires_grad()) {
        std::cout << "Creating BCELossNode (predicted requires gradients)" << std::endl;
        auto node = std::make_shared<BCELossNode<T>>(predicted, target, output);
        ComputationGraph::getInstance().addNode(node);
        std::cout << "Added node to computation graph" << std::endl;
        
        // Verify output tensor state after node creation
        std::cout << "Output tensor state after node creation:" << std::endl;
        std::cout << "  requires_grad: " << output.requires_grad() << std::endl;
        std::cout << "  gradient size: " << output.grad().size() << std::endl;
        std::cout << "  gradient value: " << output.grad()[0] << std::endl;
    } else {
        std::cout << "Skipping BCELossNode creation (predicted does not require gradients)" << std::endl;
    }
    
    // Compute forward pass
    const auto& p = predicted.data();
    const auto& t = target.data();
    auto& out = output.data();
    out.resize(1);  // Ensure output data is sized correctly
    
    // Clip predicted values to avoid log(0) and compute loss
    T total_loss = T(0);
    for (size_t i = 0; i < p.size(); ++i) {
        T pred = std::max(std::min(p[i], T(1) - T(1e-7)), T(1e-7));
        total_loss -= t[i] * std::log(pred) + (T(1) - t[i]) * std::log(T(1) - pred);
    }
    out[0] = total_loss / p.size();
    
    // Final verification of output tensor state
    std::cout << "Final output tensor state:" << std::endl;
    std::cout << "  requires_grad: " << output.requires_grad() << std::endl;
    std::cout << "  gradient size: " << output.grad().size() << std::endl;
    std::cout << "  gradient value: " << output.grad()[0] << std::endl;
    std::cout << "  loss value: " << out[0] << std::endl;
    
    std::cout << "=== binary_cross_entropy forward completed ===" << std::endl;
    return output;
}

template<typename T>
Tensor<T> mse_loss(const Tensor<T>& predicted, const Tensor<T>& target) {
    Tensor<T> output(std::vector<size_t>{1, 1});
    output.set_requires_grad(true);
    
    const auto& p = predicted.data();
    const auto& t = target.data();
    auto& out = output.data();
    
    T total_loss = T(0);
    for (size_t i = 0; i < p.size(); ++i) {
        T diff = p[i] - t[i];
        total_loss += diff * diff;
    }
    out[0] = total_loss / static_cast<T>(p.size());
    
    if (predicted.requires_grad()) {
        auto node = std::make_shared<MSELossNode<T>>(predicted, target, output);
        ComputationGraph::getInstance().addNode(node);
    }
    
    return output;
}

} // namespace ops
} // namespace dl
