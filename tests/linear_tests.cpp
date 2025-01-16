#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "dl/dl.hpp"
#include <iostream>
#include <memory>

using namespace dl::ops;

TEST_CASE("Linear layer operations") {
    dl::ComputationGraph::getInstance().clear();  // Clear graph before test
    std::shared_ptr<dl::nn::Linear<float>> layer = std::make_shared<dl::nn::Linear<float>>(2, 3);  // 2 input features, 3 output features
    
    SUBCASE("Forward pass shape") {
        dl::Tensor<float> input({4, 2});  // Batch size 4, 2 features
        auto output = layer->forward(input);
        
        CHECK_EQ(output->shape().size(), 2);
        CHECK_EQ(output->shape()[0], 4);  // Batch size preserved
        CHECK_EQ(output->shape()[1], 3);  // Output features
    }
    
    SUBCASE("Gradient computation") {
        dl::Tensor<float> input({1, 2});  // Single input
        input.set_requires_grad(true);
        
        // Set requires_grad on weights and bias
        layer->weights().set_requires_grad(true);
        layer->bias().set_requires_grad(true);
        
        auto& input_data = input.data();
        input_data[0] = 1.0f;
        input_data[1] = 2.0f;
        
        auto output = layer->forward(input);
        
        // Create target tensor
        dl::Tensor<float> target({1, 3});
        auto& target_data = target.data();
        target_data[0] = 1.0f;
        target_data[1] = 1.0f;
        target_data[2] = 1.0f;
        
        auto loss = dl::ops::mse_loss(output, target);
        
        // Initialize loss gradient
        loss->grad().assign(1, 1.0f);
        
        if (auto finalNode = (*loss).gradFn().lock())
        {
            dl::ComputationGraph::getInstance().backward(finalNode);
        }        
        dl::ComputationGraph::getInstance().clear();
        
        // Check that gradients are computed
        const auto& weight_grad = layer->weights().grad();
        const auto& bias_grad = layer->bias().grad();
        
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

TEST_CASE("Linear layer forward and backward") {
    dl::ComputationGraph::getInstance().clear();  // Clear graph before test
    // Create a simple linear layer
    auto linear = std::make_shared<dl::nn::Linear<float>>(2, 1);
    
    // Set weights and bias manually for predictable output
    auto& weights = linear->weights();
    auto& bias = linear->bias();
    weights.data() = {1.0f, 1.0f};  // 2 input features, 1 output
    bias.data() = {0.0f};           // Single bias term
    
    // Create input tensor
    dl::Tensor<float> input({1, 2});  // Batch size 1, 2 features
    input.data() = {2.0f, 3.0f};
    input.set_requires_grad(true);
    
    // Forward pass
    auto output = linear->forward(input);
    
    // Expected output: (2 * 1 + 3 * 1 + 0) = 5
    CHECK(output->data()[0] == doctest::Approx(5.0f));
    
    // Create target tensor
    dl::Tensor<float> target({1, 1});
    target.data() = {1.0f};  // Target value
    
    // Compute loss
    auto loss = dl::ops::mse_loss(*output, target);
    
    // Expected loss: (5 - 1)^2 / 1 = 16
    CHECK(loss->data()[0] == doctest::Approx(16.0f));
    
    // Initialize loss gradient
    loss->grad().assign(1, 1.0f);
    
    // Backward pass
    if (auto finalNode = (*loss).gradFn().lock())
    {
        dl::ComputationGraph::getInstance().backward(finalNode);
    }
    // Check gradients
    const auto& weight_grad = weights.grad();
    const auto& bias_grad = bias.grad();
    const auto& input_grad = input.grad();
    
    // Expected gradients can be calculated by hand
    CHECK(weight_grad[0] == doctest::Approx(16.0f));  // dL/dw1 = 2 * (5-1) * 2
    CHECK(weight_grad[1] == doctest::Approx(24.0f));  // dL/dw2 = 2 * (5-1) * 3
    CHECK(bias_grad[0] == doctest::Approx(8.0f));     // dL/db = 2 * (5-1)
    CHECK(input_grad[0] == doctest::Approx(8.0f));    // dL/dx1 = 2 * (5-1) * 1
    CHECK(input_grad[1] == doctest::Approx(8.0f));    // dL/dx2 = 2 * (5-1) * 1
}

TEST_CASE("Linear layer with batch processing") {
    dl::ComputationGraph::getInstance().clear();  // Clear graph before test
    // Create a linear layer: 2 inputs -> 1 output
    auto linear = std::make_shared<dl::nn::Linear<float>>(2, 1);
    
    // Set weights and bias manually
    auto& weights = linear->weights();
    auto& bias = linear->bias();
    weights.data() = {0.5f, 0.5f};  // Equal weights
    bias.data() = {0.0f};           // Zero bias
    
    // Create batch input: 2 samples, 2 features each
    dl::Tensor<float> input({2, 2});
    input.data() = {1.0f, 2.0f,    // First sample
                   3.0f, 4.0f};    // Second sample
    input.set_requires_grad(true);
    
    // Forward pass
    auto output = linear->forward(input);
    
    // Expected outputs:
    // Sample 1: 1 * 0.5 + 2 * 0.5 = 1.5
    // Sample 2: 3 * 0.5 + 4 * 0.5 = 3.5
    CHECK(output->data()[0] == doctest::Approx(1.5f));
    CHECK(output->data()[1] == doctest::Approx(3.5f));
    
    // Create target tensor
    dl::Tensor<float> target({2, 1});
    target.data() = {1.0f, 3.0f};  // Target values
    
    // Compute loss
    auto loss = dl::ops::mse_loss(*output, target);
    
    // Expected loss: ((1.5-1)^2 + (3.5-3)^2) / 2 = 0.25
    CHECK(loss->data()[0] == doctest::Approx(0.25f));
    
    // Initialize loss gradient
    loss->grad().assign(1, 1.0f);
    
    // Backward pass
    if (auto finalNode = (*loss).gradFn().lock())
    {
        dl::ComputationGraph::getInstance().backward(finalNode);
    }    
    // Check gradients
    const auto& weight_grad = weights.grad();
    const auto& bias_grad = bias.grad();
    const auto& input_grad = input.grad();
    
    // Verify gradient shapes
    CHECK(weight_grad.size() == 2);  // One gradient per weight
    CHECK(bias_grad.size() == 1);    // One gradient for bias
    CHECK(input_grad.size() == 4);   // One gradient per input feature
}

TEST_CASE("Linear layer forward pass produces correct output shape") {
    dl::ComputationGraph::getInstance().clear();  // Clear graph before test
    std::shared_ptr<dl::nn::Linear<float>> layer = std::make_shared<dl::nn::Linear<float>>(2, 3);
    dl::Tensor<float> input({4, 2});
    
    auto output = layer->forward(input);
    CHECK(output->shape()[0] == 4);
    CHECK(output->shape()[1] == 3);
    CHECK(layer->weights().shape()[0] == 3);
    CHECK(layer->weights().shape()[1] == 2);
}

TEST_CASE("Linear layer backward pass computes correct gradients") {
    dl::ComputationGraph::getInstance().clear();  // Clear graph before test
    std::shared_ptr<dl::nn::Linear<float>> layer = std::make_shared<dl::nn::Linear<float>>(2, 1);
    
    // Set weights and bias manually for predictable results
    auto& weights = layer->weights();
    weights.data() = {1.0f, 1.0f};  // 1x2 weight matrix
    weights.set_requires_grad(true);
    
    auto& bias = layer->bias();
    bias.data() = {1.0f};  // Single bias value
    bias.set_requires_grad(true);
    
    // Create input tensor
    dl::Tensor<float> input({1, 2});
    input.data() = {2.0f, 3.0f};
    input.set_requires_grad(true);
    
    // Forward pass
    auto output = layer->forward(input);
    
    // Create target and compute loss
    dl::Tensor<float> target({1, 1});
    target.data() = {10.0f};
    
    auto loss = dl::ops::mse_loss(output, target);
    
    // Initialize loss gradient
    loss->grad().assign(1, 1.0f);
    
    // Backward pass through computation graph
    if (auto finalNode = (*loss).gradFn().lock())
    {
        dl::ComputationGraph::getInstance().backward(finalNode);
    }
    
    // Check input gradients
    const auto& input_grad = input.grad();
    REQUIRE(input_grad.size() == 2);
        
    // Test backward pass with requires_grad
    dl::Tensor<float> input2({1, 2});
    input2.data() = {1.0f, 1.0f};
    input2.set_requires_grad(true);
    
    auto output2 = layer->forward(input2);
    output2->set_requires_grad(true);
    output2->grad().assign(output2->data().size(), 1.0f);
    
    if (auto finalNode = (*output2).gradFn().lock())
    {
        dl::ComputationGraph::getInstance().backward(finalNode);
    }    
    const auto& input2_grad = input2.grad();
    REQUIRE(input2_grad.size() == 2);
}
