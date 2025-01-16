#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "dl/tensor.hpp"
#include "dl/ops/activations.hpp"
#include "dl/ops/basic_ops.hpp"
#include "dl/ops/loss.hpp"

using namespace dl::ops;

TEST_CASE("Basic tensor operations") {
    dl::Tensor<float> a({2, 2});
    dl::Tensor<float> b({2, 2});
    
    auto& a_data = a.data();
    auto& b_data = b.data();
    
    a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    b_data = {5.0f, 6.0f, 7.0f, 8.0f};

    SUBCASE("Addition") {
        auto c = a + b;
        const auto& c_data = c.data();
        
        CHECK_EQ(c_data[0], 6.0f);
        CHECK_EQ(c_data[1], 8.0f);
        CHECK_EQ(c_data[2], 10.0f);
        CHECK_EQ(c_data[3], 12.0f);
    }
}

TEST_CASE("ReLU activation function") {
    dl::Tensor<float> input({2, 2});
    input.data() = {-1.0f, 0.0f, 1.0f, 2.0f};
    input.set_requires_grad(true);
    
    auto output = dl::ops::relu(input);
    const auto& output_data = output->data();
    
    CHECK(output_data[0] == doctest::Approx(0.0f));  // -1 -> 0
    CHECK(output_data[1] == doctest::Approx(0.0f));  // 0 -> 0
    CHECK(output_data[2] == doctest::Approx(1.0f));  // 1 -> 1
    CHECK(output_data[3] == doctest::Approx(2.0f));  // 2 -> 2
    
    // Test backward pass
    output->grad().assign(output->data().size(), 1.0f);

    if (auto final_node = output.gradFn().lock()) {
        dl::ComputationGraph::getInstance().backward(final_node);
    }
    
    const auto& input_grad = input.grad();
    CHECK(input_grad[0] == doctest::Approx(0.0f));  // grad = 0 for x < 0
    CHECK(input_grad[1] == doctest::Approx(0.0f));  // grad = 0 for x = 0
    CHECK(input_grad[2] == doctest::Approx(1.0f));  // grad = 1 for x > 0
    CHECK(input_grad[3] == doctest::Approx(1.0f));  // grad = 1 for x > 0
}

TEST_CASE("Sigmoid activation function") {
    dl::Tensor<float> input({2, 2});
    input.data() = {-2.0f, -1.0f, 1.0f, 2.0f};
    input.set_requires_grad(true);
    
    auto output = dl::ops::sigmoid(input);
    const auto& output_data = output->data();
    
    // Check sigmoid values with specific expected values
    CHECK(output_data[0] == doctest::Approx(0.119).epsilon(0.01));  // sigmoid(-2)
    CHECK(output_data[1] == doctest::Approx(0.269).epsilon(0.01));  // sigmoid(-1)
    CHECK(output_data[2] == doctest::Approx(0.731).epsilon(0.01));  // sigmoid(1)
    CHECK(output_data[3] == doctest::Approx(0.881).epsilon(0.01));  // sigmoid(2)
    
    // Test backward pass
    output->grad().assign(output->data().size(), 1.0f);

    if (auto final_node = output.gradFn().lock()) {
        dl::ComputationGraph::getInstance().backward(final_node);
    }
    
    const auto& input_grad = input.grad();
    CHECK(input_grad.size() == 4);
}

TEST_CASE("Binary Cross Entropy Loss") {
    dl::Tensor<float> predicted({1, 4});
    predicted.data() = {0.2f, 0.4f, 0.6f, 0.8f};
    predicted.set_requires_grad(true);
    auto pred_ptr = std::make_shared<dl::Tensor<float>>(predicted);
    
    dl::Tensor<float> target({1, 4});
    target.data() = {0.0f, 0.0f, 1.0f, 1.0f};
    
    auto loss = dl::ops::binary_cross_entropy(pred_ptr, target);
    CHECK(loss->data()[0] > 0.0f);  // Loss should be positive
    
    // Test backward pass
    loss->grad().assign(1, 1.0f);
    if (auto final_node = loss.gradFn().lock()) {
        dl::ComputationGraph::getInstance().backward(final_node);
    }
    
    const auto& pred_grad = predicted.grad();
    CHECK(pred_grad.size() == 4);
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
    
    auto loss = dl::ops::mse_loss(pred, target);
    const auto& loss_data = loss->data();
    
    // MSE = ((0.5 - 1.0)^2 + (0.8 - 0.0)^2) / 2 = (0.25 + 0.64) / 2 = 0.445
    CHECK_EQ(loss_data[0], doctest::Approx(0.445f));
}
