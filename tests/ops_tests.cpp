#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "dl/tensor.hpp"
#include "dl/ops/activations.hpp"
#include "dl/ops/basic_ops.hpp"
#include "dl/ops/loss.hpp"

using namespace dl::ops;

TEST_CASE("Basic operations") {
    dl::Tensor<float> a({2, 2});
    dl::Tensor<float> b({2, 2});
    
    auto& a_data = a.data();
    auto& b_data = b.data();
    
    // Initialize tensors
    a_data[0] = 1.0f; a_data[1] = 2.0f;
    a_data[2] = 3.0f; a_data[3] = 4.0f;
    
    b_data[0] = 5.0f; b_data[1] = 6.0f;
    b_data[2] = 7.0f; b_data[3] = 8.0f;

    SUBCASE("Addition") {
        auto c = a + b;
        const auto& c_data = c.data();
        
        CHECK_EQ(c_data[0], 6.0f);
        CHECK_EQ(c_data[1], 8.0f);
        CHECK_EQ(c_data[2], 10.0f);
        CHECK_EQ(c_data[3], 12.0f);
    }
}

TEST_CASE("ReLU activation") {
    dl::Tensor<float> input({4});
    auto& input_data = input.data();
    
    input_data[0] = -2.0f;
    input_data[1] = -1.0f;
    input_data[2] = 1.0f;
    input_data[3] = 2.0f;
    
    auto output = relu(input);
    const auto& output_data = output.data();
    
    CHECK_EQ(output_data[0], 0.0f);
    CHECK_EQ(output_data[1], 0.0f);
    CHECK_EQ(output_data[2], 1.0f);
    CHECK_EQ(output_data[3], 2.0f);
}

TEST_CASE("Sigmoid activation") {
    dl::Tensor<float> input({3});
    auto& input_data = input.data();
    
    input_data[0] = 0.0f;
    input_data[1] = 1.0f;
    input_data[2] = -1.0f;
    
    auto output = sigmoid(input);
    const auto& output_data = output.data();
    
    CHECK_EQ(output_data[0], doctest::Approx(0.5f));
    CHECK_EQ(output_data[1], doctest::Approx(0.731059f));
    CHECK_EQ(output_data[2], doctest::Approx(0.268941f));
}

TEST_CASE("Loss functions") {
    dl::Tensor<float> pred({2});
    dl::Tensor<float> target({2});
    
    auto& pred_data = pred.data();
    auto& target_data = target.data();
    
    pred_data[0] = 0.5f;
    pred_data[1] = 0.8f;
    
    target_data[0] = 1.0f;
    target_data[1] = 0.0f;
    
    auto loss = mse_loss(pred, target);
    const auto& loss_data = loss.data();
    
    // MSE = ((0.5 - 1.0)^2 + (0.8 - 0.0)^2) / 2 = (0.25 + 0.64) / 2 = 0.445
    CHECK_EQ(loss_data[0], doctest::Approx(0.445f));
}
