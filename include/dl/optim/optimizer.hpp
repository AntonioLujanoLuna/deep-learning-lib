#pragma once

#include "../tensor.hpp"
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace dl {
namespace optim {

template<typename T>
class SGD {
public:
    SGD(T learning_rate) : learning_rate_(learning_rate) {
        if (learning_rate <= T(0)) {
            throw std::runtime_error("Learning rate must be positive");
        }
    }

    void add_parameter(Tensor<T>& param) {
        if (!param.requires_grad()) {
            std::cout << "Setting requires_grad=true for parameter tensor" << std::endl;
            param.set_requires_grad(true);
        }
        parameters_.push_back(&param);
    }

    void add_parameters(const std::vector<std::reference_wrapper<Tensor<T>>>& params) {
        for (auto& param : params) {
            add_parameter(param.get());
        }
    }

    void zero_grad() {
        std::cout << "Zeroing gradients for " << parameters_.size() << " parameters" << std::endl;
        for (auto& param : parameters_) {
            param->zero_grad();
        }
    }

    void step() {
        static int step_count = 0;
        bool print_debug = step_count < 5;  // Print for first 5 steps
        
        if (print_debug) {
            std::cout << "\n=== SGD Step " << step_count << " ===" << std::endl;
            std::cout << "Learning rate: " << learning_rate_ << std::endl;
            std::cout << "Number of parameters: " << parameters_.size() << std::endl;
        }
        
        for (auto& param : parameters_) {
            if (!param->requires_grad()) {
                std::cout << "Warning: Parameter tensor does not require gradients" << std::endl;
                continue;
            }
            
            const auto& grad = param->grad();
            auto& data = param->data();
            
            // Validate tensor sizes
            if (grad.size() != data.size()) {
                throw std::runtime_error("Gradient size (" + std::to_string(grad.size()) + 
                                       ") does not match parameter size (" + std::to_string(data.size()) + ")");
            }
            
            if (print_debug) {
                std::cout << "\nParameter update:" << std::endl;
                std::cout << "Shape: " << utils::shape_to_string(param->shape()) << std::endl;
                
                // Print statistics
                T max_grad = std::numeric_limits<T>::lowest();
                T min_grad = std::numeric_limits<T>::max();
                T avg_grad = T(0);
                
                for (const auto& g : grad) {
                    max_grad = std::max(max_grad, g);
                    min_grad = std::min(min_grad, g);
                    avg_grad += g;
                }
                avg_grad /= grad.size();
                
                std::cout << "Gradient stats - Min: " << min_grad 
                          << ", Max: " << max_grad 
                          << ", Avg: " << avg_grad << std::endl;
                
                // Print first few values
                std::cout << "First 3 values before update:" << std::endl;
                for (size_t i = 0; i < std::min(size_t(3), data.size()); ++i) {
                    std::cout << "  data[" << i << "] = " << data[i] 
                             << ", grad[" << i << "] = " << grad[i] << std::endl;
                }
            }
            
            // Update parameters with gradient clipping
            const T clip_value = T(1);
            for (size_t i = 0; i < data.size(); ++i) {
                T clipped_grad = std::max(std::min(grad[i], clip_value), -clip_value);
                data[i] -= learning_rate_ * clipped_grad;
            }
            
            if (print_debug) {
                std::cout << "First 3 values after update:" << std::endl;
                for (size_t i = 0; i < std::min(size_t(3), data.size()); ++i) {
                    std::cout << "  data[" << i << "] = " << data[i] << std::endl;
                }
            }
        }
        
        step_count++;
    }

private:
    T learning_rate_;
    std::vector<Tensor<T>*> parameters_;
};

} // namespace optim
} // namespace dl
