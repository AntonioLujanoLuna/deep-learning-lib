#pragma once

#include "../tensor.hpp"
#include <vector>
#include <memory>

namespace dl {
namespace optim {

template<typename T>
class Optimizer {
public:
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual ~Optimizer() = default;

    void add_parameter(Tensor<T>& param) {
        parameters_.push_back(&param);
    }

protected:
    std::vector<Tensor<T>*> parameters_;
};

template<typename T>
class SGD : public Optimizer<T> {
public:
    SGD(T learning_rate = T(0.01), T momentum = T(0))
        : lr_(learning_rate), momentum_(momentum) {
        if (momentum_ > 0) {
            velocities_.resize(this->parameters_.size());
        }
    }

    void step() override {
        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto& param = *this->parameters_[i];
            auto& param_data = param.data();
            const auto& grad = param.grad();

            if (momentum_ > 0) {
                if (velocities_[i].empty()) {
                    velocities_[i].resize(grad.size());
                }
                
                for (size_t j = 0; j < grad.size(); ++j) {
                    velocities_[i][j] = momentum_ * velocities_[i][j] + lr_ * grad[j];
                    param_data[j] -= velocities_[i][j];
                }
            } else {
                for (size_t j = 0; j < grad.size(); ++j) {
                    param_data[j] -= lr_ * grad[j];
                }
            }
        }
    }

    void zero_grad() override {
        for (auto& param : this->parameters_) {
            param->zero_grad();
        }
    }

private:
    T lr_;
    T momentum_;
    std::vector<std::vector<T>> velocities_;
};

template<typename T>
class Adam : public Optimizer<T> {
public:
    Adam(T learning_rate = T(0.001), T beta1 = T(0.9), T beta2 = T(0.999), T epsilon = T(1e-8))
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

    void step() override {
        if (m_.empty()) {
            m_.resize(this->parameters_.size());
            v_.resize(this->parameters_.size());
        }

        t_++;
        T alpha = lr_ * std::sqrt(T(1) - std::pow(beta2_, t_)) / (T(1) - std::pow(beta1_, t_));

        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto& param = *this->parameters_[i];
            auto& param_data = param.data();
            const auto& grad = param.grad();

            if (m_[i].empty()) {
                m_[i].resize(grad.size());
                v_[i].resize(grad.size());
            }

            for (size_t j = 0; j < grad.size(); ++j) {
                m_[i][j] = beta1_ * m_[i][j] + (T(1) - beta1_) * grad[j];
                v_[i][j] = beta2_ * v_[i][j] + (T(1) - beta2_) * grad[j] * grad[j];
                
                param_data[j] -= alpha * m_[i][j] / (std::sqrt(v_[i][j]) + epsilon_);
            }
        }
    }

    void zero_grad() override {
        for (auto& param : this->parameters_) {
            param->zero_grad();
        }
    }

private:
    T lr_;
    T beta1_;
    T beta2_;
    T epsilon_;
    size_t t_;
    std::vector<std::vector<T>> m_; // First moment
    std::vector<std::vector<T>> v_; // Second moment
};

} // namespace optim
} // namespace dl
