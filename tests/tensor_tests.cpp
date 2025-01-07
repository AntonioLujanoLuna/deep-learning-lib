#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "dl/tensor.hpp"

TEST_CASE("Tensor basic operations") {
    dl::Tensor<float> tensor({2, 3}); // Create a 2x3 tensor

    SUBCASE("Tensor creation") {
        CHECK_EQ(tensor.shape()[0], 2);
        CHECK_EQ(tensor.shape()[1], 3);
    }

    SUBCASE("Gradient operations") {
        tensor.zero_grad();
        // Verify grad is zeroed
        const auto& grad = tensor.grad();
        for (const auto& g : grad) {
            CHECK_EQ(g, 0.0f);
        }
    }
}

TEST_CASE("Tensor arithmetic operations") {
    dl::Tensor<float> a({2, 2});
    dl::Tensor<float> b({2, 2});
    
    // Initialize tensors with some values
    auto& a_data = a.data();
    auto& b_data = b.data();
    
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f; a_data[3] = 4.0f;
    b_data[0] = 2.0f; b_data[1] = 3.0f; b_data[2] = 4.0f; b_data[3] = 5.0f;

    // Enable gradients
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    SUBCASE("Addition") {
        auto c = a + b;
        const auto& c_data = c.data();
        CHECK_EQ(c_data[0], 3.0f);
        CHECK_EQ(c_data[1], 5.0f);
        CHECK_EQ(c_data[2], 7.0f);
        CHECK_EQ(c_data[3], 9.0f);
    }

    SUBCASE("Multiplication") {
        auto c = a * b;
        const auto& c_data = c.data();
        CHECK_EQ(c_data[0], 2.0f);
        CHECK_EQ(c_data[1], 6.0f);
        CHECK_EQ(c_data[2], 12.0f);
        CHECK_EQ(c_data[3], 20.0f);
    }

    SUBCASE("Autograd - Addition") {
        a.zero_grad();
        b.zero_grad();
        auto c = a + b;
        c.backward();
        
        // Gradients should flow equally to both inputs
        CHECK_EQ(a.grad()[0], 1.0f);
        CHECK_EQ(b.grad()[0], 1.0f);
    }

    SUBCASE("Autograd - Multiplication") {
        a.zero_grad();
        b.zero_grad();
        auto c = a * b;
        c.backward();
        
        // Gradient of a*b with respect to a is b
        CHECK_EQ(a.grad()[0], b.data()[0]);
        // Gradient of a*b with respect to b is a
        CHECK_EQ(b.grad()[0], a.data()[0]);
    }
}
