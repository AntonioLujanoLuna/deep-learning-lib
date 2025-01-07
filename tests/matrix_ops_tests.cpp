#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "dl/tensor.hpp"
#include "dl/ops/matrix_ops.hpp"

TEST_CASE("Matrix operations") {
    dl::Tensor<float> a({2, 3});
    dl::Tensor<float> b({3, 2});

    SUBCASE("Matrix multiplication shape check") {
        auto c = dl::ops::matmul(a, b);
        CHECK_EQ(c.shape()[0], 2);
        CHECK_EQ(c.shape()[1], 2);
    }

    SUBCASE("Matrix multiplication computation") {
        // Initialize matrices
        auto& a_data = a.data();
        auto& b_data = b.data();
        
        // Matrix A:
        // 1 2 3
        // 4 5 6
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
        a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;
        
        // Matrix B:
        // 1 2
        // 3 4
        // 5 6
        b_data[0] = 1.0f; b_data[1] = 2.0f;
        b_data[2] = 3.0f; b_data[3] = 4.0f;
        b_data[4] = 5.0f; b_data[5] = 6.0f;

        auto c = dl::ops::matmul(a, b);
        const auto& c_data = c.data();
        
        // Expected result:
        // [1*1 + 2*3 + 3*5  1*2 + 2*4 + 3*6]
        // [4*1 + 5*3 + 6*5  4*2 + 5*4 + 6*6]
        // = [22 28]
        //   [49 64]
        CHECK_EQ(c_data[0], 22.0f);
        CHECK_EQ(c_data[1], 28.0f);
        CHECK_EQ(c_data[2], 49.0f);
        CHECK_EQ(c_data[3], 64.0f);
    }

    SUBCASE("Matrix multiplication gradient") {
        // Enable gradients
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        
        // Initialize matrices
        auto& a_data = a.data();
        auto& b_data = b.data();
        
        // Matrix A:
        // 1 2 3
        // 4 5 6
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
        a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;
        
        // Matrix B:
        // 1 2
        // 3 4
        // 5 6
        b_data[0] = 1.0f; b_data[1] = 2.0f;
        b_data[2] = 3.0f; b_data[3] = 4.0f;
        b_data[4] = 5.0f; b_data[5] = 6.0f;

        // Zero gradients before forward pass
        a.zero_grad();
        b.zero_grad();

        auto c = dl::ops::matmul(a, b);
        c.backward();

        // Check gradients
        const auto& a_grad = a.grad();
        const auto& b_grad = b.grad();

        // Verify gradients are non-zero
        bool has_nonzero_grad = false;
        for (const auto& g : a_grad) {
            if (g != 0.0f) {
                has_nonzero_grad = true;
                break;
            }
        }
        CHECK(has_nonzero_grad);

        has_nonzero_grad = false;
        for (const auto& g : b_grad) {
            if (g != 0.0f) {
                has_nonzero_grad = true;
                break;
            }
        }
        CHECK(has_nonzero_grad);
    }
}
