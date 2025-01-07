#pragma once

#include "../tensor.hpp"
#include <vector>
#include <iostream>

namespace dl {
namespace optim {

template<typename T>
class SGD {
public:
    SGD(T learning_rate) : learning_rate_(learning_rate) {}

    void add_parameter(Tensor<T>& param) {
        if (!param.requires_grad()) {
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
        for (auto& param : parameters_) {
            param->zero_grad();
        }
    }

    void step() {
        static int step_count = 0;
        bool print_debug = step_count < 2;  // Only print for first two steps
        
        if (print_debug) {
            std::cout << "SGD Step " << step_count << ":" << std::endl;
        }
        
        for (auto& param : parameters_) {
            const auto& grad = param->grad();
            auto& data = param->data();
            
            if (print_debug) {
                // Print first few values
                std::cout << "Parameter data (first 3): ";
                for (size_t i = 0; i < std::min(size_t(3), data.size()); ++i) {
                    std::cout << data[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "Gradients (first 3): ";
                for (size_t i = 0; i < std::min(size_t(3), grad.size()); ++i) {
                    std::cout << grad[i] << " ";
                }
                std::cout << std::endl;
            }
            
            // Update parameters
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] -= learning_rate_ * grad[i];
            }
            
            if (print_debug) {
                std::cout << "Updated data (first 3): ";
                for (size_t i = 0; i < std::min(size_t(3), data.size()); ++i) {
                    std::cout << data[i] << " ";
                }
                std::cout << std::endl << std::flush;
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
