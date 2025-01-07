#include "dl/tensor.hpp"
#include "dl/autograd.hpp"
#include "dl/ops/broadcast.hpp"
#include "dl/ops/basic_ops.hpp"

namespace dl {
// Explicit template instantiation for float
template class Tensor<float>;
} // namespace dl