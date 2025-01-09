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
        //std::cout << "Output requires_grad: " << output.requires_grad() << std::endl;
        
        // Create target tensor
        dl::Tensor<float> target({1, 3});
        auto& target_data = target.data();
        target_data[0] = 1.0f;
        target_data[1] = 1.0f;
        target_data[2] = 1.0f;
        
        auto loss = dl::ops::mse_loss(output, target);
        
        dl::ComputationGraph::getInstance().backward();
        dl::ComputationGraph::getInstance().clear();
        
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

TEST_CASE("Linear layer backward pass") {
    dl::nn::Linear<float> layer(2, 3);  // 2 inputs, 3 outputs
    
    // Set weights and bias for testing
    auto& weights = layer.weights().data();
    weights = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};  // 2x3 matrix
    
    auto& bias = layer.bias().data();
    bias = {0.1f, 0.2f, 0.3f};  // 3 elements
    
    // Create input tensor
    dl::Tensor<float> input({1, 2});
    input.data() = {1.0f, 2.0f};
    input.set_requires_grad(true);
    
    // Forward pass
    auto output = layer.forward(input);
    output.set_requires_grad(true);
    
    // Set output gradient
    output.grad() = {1.0f, 1.0f, 1.0f};
    
    // Backward pass
    dl::ComputationGraph::getInstance().backward();
    dl::ComputationGraph::getInstance().clear();
    
    // Check input gradients
    std::vector<float> expected_input_grad = {0.6f, 1.5f};  // dot product with weights
    for (size_t i = 0; i < input.grad().size(); ++i) {
        CHECK(input.grad()[i] == doctest::Approx(expected_input_grad[i]));
    }
    
    // Check weight gradients
    std::vector<float> expected_weight_grad = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    for (size_t i = 0; i < layer.weights().grad().size(); ++i) {
        CHECK(layer.weights().grad()[i] == doctest::Approx(expected_weight_grad[i]));
    }
    
    // Check bias gradients
    std::vector<float> expected_bias_grad = {1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < layer.bias().grad().size(); ++i) {
        CHECK(layer.bias().grad()[i] == doctest::Approx(expected_bias_grad[i]));
    }
}
