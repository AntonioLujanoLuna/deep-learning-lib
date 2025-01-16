#pragma once

#include <memory>
#include <vector>

namespace dl {
    
// Forward declare the base Node class
class Node;

// Forward declare the ComputationGraph
class ComputationGraph;

// Forward declare the namespace detail and TensorImpl
namespace detail {
    template<typename T>
    class TensorImpl;
}

// Forward declare Tensor
template<typename T>
class Tensor;

// Forward declarations for Neural Network components
namespace nn {
    template<typename T>
    class Linear;

    template<typename T>
    class MLP;
}

// Forward declarations for Optimizers
namespace optim {
    template<typename T>
    class SGD;
}

// Forward declarations for Operations
namespace ops {
    template<typename T> class AddNode;
    template<typename T> class MulNode;
    template<typename T> class MatMulNode;
    template<typename T> class ReLUNode;
    template<typename T> class SigmoidNode;
    template<typename T> class TanhNode;
    template<typename T> class SumNode;
    template<typename T> class MeanNode;
    template<typename T> class MaxNode;
    template<typename T> class MinNode;
    template<typename T> class ProdNode;
    template<typename T> class Conv2DNode;
    template<typename T> class MSELossNode;
    template<typename T> class BCELossNode;
} // namespace ops

} // namespace dl
