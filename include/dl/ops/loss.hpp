#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "dl/tensor.hpp"
#include "dl/autograd.hpp"

namespace dl {
namespace ops {

// Non-template version of shape_to_string
inline std::string shape_to_string(const std::vector<size_t>& shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

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
        , output_size_(std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies<size_t>())) {
        
        std::cout << "BCELossNode constructor:" << std::endl;
        std::cout << "  Predicted shape: " << shape_to_string(predicted_shape_) << std::endl;
        std::cout << "  Target shape: " << shape_to_string(target_shape_) << std::endl;
        std::cout << "  Output shape: " << shape_to_string(output_shape_) << std::endl;
        std::cout << "  Predicted requires grad: " << std::boolalpha << predicted_.requires_grad() << std::endl;
        std::cout << "  Predicted grad size: " << predicted_.grad().size() << std::endl;
        
        if (predicted_.shape().empty() || target_.shape().empty() || output_.shape().empty() || output_.shape() != std::vector<size_t>{1, 1}) {
            throw std::runtime_error("Invalid tensor shapes in BCELossNode. Predicted shape: " + shape_to_string(predicted_.shape()) + ", Target shape: " + shape_to_string(target_.shape()) + ", Output shape: " + shape_to_string(output_.shape()));
        }
    }

    void backward() override {
        std::cout << "\n=== Starting BCE Loss backward ===" << std::endl;
        
        try {
            // Debug point 1
            std::cout << "Debug point 1: Getting shapes" << std::endl;
            std::cout << "Input shape: " << shape_to_string(predicted_shape_) << std::endl;
            std::cout << "Target shape: " << shape_to_string(target_shape_) << std::endl;
            std::cout << "Output shape: " << shape_to_string(output_shape_) << std::endl;

            // Debug point 2
            std::cout << "Debug point 2: Getting tensors" << std::endl;
            if (!predicted_.requires_grad()) {
                std::cout << "Warning: Predicted tensor does not require gradients" << std::endl;
            }

            // Get all tensors first
            std::cout << "Getting output gradient..." << std::endl;
            const auto& output_grad = output_.grad();
            std::cout << "Output gradient size: " << output_grad.size() << std::endl;
            std::cout << "Output gradient value: " << output_grad[0] << std::endl;

            std::cout << "Getting predicted gradient..." << std::endl;
            auto& predicted_grad = const_cast<Tensor<T>&>(predicted_).grad();
            std::cout << "Predicted gradient size: " << predicted_grad.size() << std::endl;

            std::cout << "Getting predicted data..." << std::endl;
            const auto& predicted_data = predicted_.data();
            std::cout << "Predicted data size: " << predicted_data.size() << std::endl;

            std::cout << "Getting target data..." << std::endl;
            const auto& target_data = target_.data();
            std::cout << "Target data size: " << target_data.size() << std::endl;

            // Debug point 3
            std::cout << "Debug point 3: Validating sizes" << std::endl;
            std::cout << "Expected sizes - Output: " << output_size_ 
                      << ", Predicted: " << predicted_size_
                      << ", Target: " << target_size_ << std::endl;
            
            std::cout << "Actual sizes - Output grad: " << output_grad.size()
                      << ", Predicted grad: " << predicted_grad.size()
                      << ", Predicted data: " << predicted_data.size()
                      << ", Target data: " << target_data.size() << std::endl;

            // Validate sizes with detailed error messages
            if (output_grad.size() != output_size_) {
                throw std::runtime_error("Output gradient size mismatch. Expected: " + std::to_string(output_size_) + ", Got: " + std::to_string(output_grad.size()));
            }
            if (predicted_grad.size() != predicted_size_) {
                throw std::runtime_error("Predicted gradient size mismatch. Expected: " + std::to_string(predicted_size_) + ", Got: " + std::to_string(predicted_grad.size()));
            }
            if (predicted_data.size() != predicted_size_) {
                throw std::runtime_error("Predicted data size mismatch. Expected: " + std::to_string(predicted_size_) + ", Got: " + std::to_string(predicted_data.size()));
            }
            if (target_data.size() != target_size_) {
                throw std::runtime_error("Target data size mismatch. Expected: " + std::to_string(target_size_) + ", Got: " + std::to_string(target_data.size()));
            }

            // Debug point 4
            std::cout << "Debug point 4: Starting gradient computation" << std::endl;
            std::cout << "Output gradient value: " << output_grad[0] << std::endl;
            
            // Print first few values with more detail
            std::cout << "Input tensor values (first 3):" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < std::min(size_t(3), predicted_data.size()); ++i) {
                std::cout << "Index " << i << ":"
                         << " pred=" << predicted_data[i]
                         << " target=" << target_data[i]
                         << " current_grad=" << predicted_grad[i] << std::endl;
            }

            // Debug point 5
            std::cout << "Debug point 5: Computing gradients" << std::endl;
            
            // Compute gradients with detailed logging
            for (size_t i = 0; i < predicted_data.size(); ++i) {
                T p = predicted_data[i];
                T t = target_data[i];
                
                // Clip values to avoid division by zero
                p = std::max(std::min(p, T(1) - T(1e-7)), T(1e-7));
                
                // Compute gradient components separately for debugging
                T term1 = -t/p;
                T term2 = (T(1)-t)/(T(1)-p);
                T grad_before_scale = term1 + term2;
                T final_grad = grad_before_scale * output_grad[0];
                
                predicted_grad[i] = final_grad;
                
                if (i < 3) {  // Print detailed computation for first few elements
                    std::cout << "Gradient computation for index " << i << ":" << std::endl;
                    std::cout << "  p = " << p << ", t = " << t << std::endl;
                    std::cout << "  term1 (-t/p) = " << term1 << std::endl;
                    std::cout << "  term2 ((1-t)/(1-p)) = " << term2 << std::endl;
                    std::cout << "  grad before scale = " << grad_before_scale << std::endl;
                    std::cout << "  final grad = " << final_grad << std::endl;
                }
            }

            // Debug point 6
            std::cout << "Debug point 6: Final gradients" << std::endl;
            std::cout << "First few gradients after computation:" << std::endl;
            for (size_t i = 0; i < std::min(size_t(3), predicted_grad.size()); ++i) {
                std::cout << "grad[" << i << "] = " << predicted_grad[i] << std::endl;
            }
            
            std::cout << "=== BCE Loss backward completed ===" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "\nError in BCELossNode backward: " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "\nUnknown error in BCELossNode backward" << std::endl;
            throw;
        }
    }

private:
    const Tensor<T>& predicted_;
    const Tensor<T>& target_;
    Tensor<T>& output_;
    const std::vector<size_t> predicted_shape_;
    const std::vector<size_t> target_shape_;
    const std::vector<size_t> output_shape_;
    const size_t predicted_size_;
    const size_t target_size_;
    const size_t output_size_;
};

template<typename T>
class MSELossNode : public Node {
public:
    MSELossNode(const Tensor<T>& pred, const Tensor<T>& target, Tensor<T>& loss)
        : pred_(pred), target_(target), loss_(loss) {
        if (pred_.shape().empty() || target_.shape().empty() || loss_.shape().empty() || loss_.shape() != std::vector<size_t>{1, 1}) {
            throw std::runtime_error("Invalid tensor shapes in MSELossNode. Predicted shape: " + shape_to_string(pred_.shape()) + ", Target shape: " + shape_to_string(target_.shape()) + ", Loss shape: " + shape_to_string(loss_.shape()));
        }
        
        // Store shapes for later use
        pred_shape_ = pred_.shape();
        target_shape_ = target_.shape();
        loss_shape_ = loss_.shape();
        
        // Store initial sizes
        pred_size_ = std::accumulate(pred_shape_.begin(), pred_shape_.end(), size_t(1), std::multiplies<size_t>());
        target_size_ = std::accumulate(target_shape_.begin(), target_shape_.end(), size_t(1), std::multiplies<size_t>());
        loss_size_ = std::accumulate(loss_shape_.begin(), loss_shape_.end(), size_t(1), std::multiplies<size_t>());
    }

    void backward() override {
        if (pred_.requires_grad()) {
            auto& pred_grad = const_cast<Tensor<T>&>(pred_).grad();
            const auto& pred_data = pred_.data();
            const auto& target_data = target_.data();
            const auto& loss_grad = loss_.grad();
            
            // Validate sizes
            if (pred_grad.size() != pred_size_) {
                throw std::runtime_error("Predicted gradient size mismatch. Expected: " + std::to_string(pred_size_) + ", Got: " + std::to_string(pred_grad.size()));
            }
            if (pred_data.size() != pred_size_) {
                throw std::runtime_error("Predicted data size mismatch. Expected: " + std::to_string(pred_size_) + ", Got: " + std::to_string(pred_data.size()));
            }
            if (target_data.size() != target_size_) {
                throw std::runtime_error("Target data size mismatch. Expected: " + std::to_string(target_size_) + ", Got: " + std::to_string(target_data.size()));
            }
            if (loss_grad.size() != loss_size_) {
                throw std::runtime_error("Loss gradient size mismatch. Expected: " + std::to_string(loss_size_) + ", Got: " + std::to_string(loss_grad.size()));
            }
            
            T scale = T(2) / pred_data.size();
            for (size_t i = 0; i < pred_data.size(); ++i) {
                pred_grad[i] += scale * (pred_data[i] - target_data[i]) * loss_grad[0];
            }
        }
    }

private:
    const Tensor<T>& pred_;
    const Tensor<T>& target_;
    Tensor<T>& loss_;
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
    
    // Create output tensor
    Tensor<T> output(std::vector<size_t>{1, 1});
    std::cout << "Created output tensor with shape: " << shape_to_string(output.shape()) << std::endl;
    
    // Set requires_grad before initializing gradient
    output.set_requires_grad(true);
    std::cout << "Output requires grad: " << std::boolalpha << output.requires_grad() << std::endl;
    
    // Initialize output gradient to 1
    output.grad()[0] = T(1);
    std::cout << "Output gradient size: " << output.grad().size() << std::endl;
    std::cout << "Output gradient value: " << output.grad()[0] << std::endl;
    
    // Verify gradient is still set
    std::cout << "Verifying output gradient before creating node:" << std::endl;
    std::cout << "Output gradient size: " << output.grad().size() << std::endl;
    std::cout << "Output gradient value: " << output.grad()[0] << std::endl;
    
    // Create and add node to computation graph
    if (predicted.requires_grad()) {
        auto node = std::make_shared<BCELossNode<T>>(predicted, target, output);
        ComputationGraph::getInstance().addNode(node);
    }
    
    // Compute forward pass
    const auto& p = predicted.data();
    const auto& t = target.data();
    auto& out = output.data();
    
    T total_loss = T(0);
    for (size_t i = 0; i < p.size(); ++i) {
        // Clip values to avoid log(0)
        T p_i = std::max(std::min(p[i], T(1) - T(1e-7)), T(1e-7));
        T t_i = t[i];
        
        // Standard binary cross entropy formula: -t*log(p) - (1-t)*log(1-p)
        total_loss += -t_i * std::log(p_i) - (T(1) - t_i) * std::log(T(1) - p_i);
    }
    out[0] = total_loss / static_cast<T>(p.size());

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
