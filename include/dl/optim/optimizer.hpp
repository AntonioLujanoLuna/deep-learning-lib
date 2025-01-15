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
        parameters_.push_back(param);
    }

    void add_parameters(const std::vector<std::reference_wrapper<Tensor<T>>>& params) {
        for (auto& param : params) {
            add_parameter(param.get());
        }
    }

    void zero_grad() {
        for (auto& param : parameters_) {
            param.get().zero_grad();
        }
    }

    void step() {
        for (auto& param : parameters_) {
            if (!param.get().requires_grad()) {
                std::cout << "Warning: Parameter tensor does not require gradients" << std::endl;
                continue;
            }
            
            const auto& grad = param.get().grad();
            auto& data = param.get().data();
            
            // Validate tensor sizes
            if (grad.size() != data.size()) {
                throw std::runtime_error("Gradient size (" + std::to_string(grad.size()) + 
                                       ") does not match parameter size (" + std::to_string(data.size()) + ")");
            }
            
            // Update parameters with gradient clipping
            const T clip_value = T(1);  // Reduced from 5 to 1 for more stable training
            T max_norm = T(0);
            
            // First compute gradient norm
            for (size_t i = 0; i < grad.size(); ++i) {
                max_norm = std::max(max_norm, std::abs(grad[i]));
            }
            
            // Scale gradients if norm is too large
            T scale = max_norm > clip_value ? clip_value / max_norm : T(1);
            
            // Apply updates
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] -= learning_rate_ * grad[i] * scale;
            }
        }
    }

private:
    T learning_rate_;
    std::vector<std::reference_wrapper<Tensor<T>>> parameters_;
};

} // namespace optim
} // namespace dl
