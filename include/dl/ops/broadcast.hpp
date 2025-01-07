#pragma once

#include "../tensor.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dl {
namespace ops {

inline std::vector<size_t> compute_broadcast_shape(
    const std::vector<size_t>& shape1,
    const std::vector<size_t>& shape2) {
    
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<size_t> result(max_dims);
    
    // Pad shorter shape with leading 1s
    auto pad_shape = [max_dims](const std::vector<size_t>& shape) {
        std::vector<size_t> padded(max_dims, 1);
        std::copy(shape.rbegin(), shape.rend(), padded.rbegin());
        return padded;
    };
    
    auto padded1 = pad_shape(shape1);
    auto padded2 = pad_shape(shape2);
    
    // Compute broadcast shape
    for (size_t i = 0; i < max_dims; ++i) {
        if (padded1[i] == padded2[i]) {
            result[i] = padded1[i];
        } else if (padded1[i] == 1) {
            result[i] = padded2[i];
        } else if (padded2[i] == 1) {
            result[i] = padded1[i];
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }
    }
    
    return result;
}

template<typename T>
std::vector<T> broadcast_to(const std::vector<T>& data,
                           const std::vector<size_t>& from_shape,
                           const std::vector<size_t>& to_shape) {
    size_t from_size = std::accumulate(from_shape.begin(), from_shape.end(), 
                                     size_t(1), std::multiplies<size_t>());
    size_t to_size = std::accumulate(to_shape.begin(), to_shape.end(), 
                                   size_t(1), std::multiplies<size_t>());
    
    std::vector<T> result(to_size);
    std::vector<size_t> from_strides(from_shape.size());
    std::vector<size_t> to_strides(to_shape.size());
    
    // Compute strides
    size_t stride = 1;
    for (int i = from_shape.size() - 1; i >= 0; --i) {
        from_strides[i] = stride;
        stride *= from_shape[i];
    }
    
    stride = 1;
    for (int i = to_shape.size() - 1; i >= 0; --i) {
        to_strides[i] = stride;
        stride *= to_shape[i];
    }
    
    // Perform broadcasting
    for (size_t i = 0; i < to_size; ++i) {
        size_t from_index = 0;
        size_t temp = i;
        
        for (size_t dim = 0; dim < to_shape.size(); ++dim) {
            size_t coord = temp / to_strides[dim];
            temp %= to_strides[dim];
            
            if (dim >= to_shape.size() - from_shape.size()) {
                size_t from_dim = dim - (to_shape.size() - from_shape.size());
                if (from_shape[from_dim] > 1) {
                    from_index += coord * from_strides[from_dim];
                }
            }
        }
        
        result[i] = data[from_index];
    }
    
    return result;
}

} // namespace ops
} // namespace dl
