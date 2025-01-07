#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "dl/nn/linear.hpp"
#include "dl/ops/loss.hpp"
#include <iostream>

TEST_CASE("Linear layer operations") {
    dl::nn::Linear<float> layer(2, 3);  // 2 input features, 3 output features
    
    SUBCASE("Forward pass shape") {
        dl::Tensor<float> input({4, 2});  // Batch size 4, 2 features
        auto output = layer.forward(input);
        
        CHECK_EQ(output.shape().size(), 2);
        CHECK_EQ(output.shape()[0], 4);  // Batch size preserved
        CHECK_EQ(output.shape()[1], 3);  // Output features
    }
    
    SUBCASE("Gradient computation") {
        dl::Tensor<float> input({1, 2});  // Single input
        input.set_requires_grad(true);
        
        auto& input_data = input.data();
        input_data[0] = 1.0f;
        input_data[1] = 2.0f;
        
        auto output = layer.forward(input);
        std::cout << "Output requires_grad: " << output.requires_grad() << std::endl;
        
        // Create target tensor
        dl::Tensor<float> target({1, 3});
        auto& target_data = target.data();
        target_data[0] = 1.0f;
        target_data[1] = 1.0f;
        target_data[2] = 1.0f;
        
        auto loss = dl::ops::mse_loss(output, target);
        
        loss.backward();
        
        // Check that gradients are computed
        const auto& weight_grad = layer.weights().grad();
        const auto& bias_grad = layer.bias().grad();
        
        bool has_weight_grad = false;
        for (const auto& g : weight_grad) {
            if (g != 0.0f) {
                has_weight_grad = true;
                break;
            }
        }
        CHECK(has_weight_grad);
        
        bool has_bias_grad = false;
        for (const auto& g : bias_grad) {
            if (g != 0.0f) {
                has_bias_grad = true;
                break;
            }
        }
        CHECK(has_bias_grad);
    }
}
