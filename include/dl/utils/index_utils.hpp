#pragma once

#include <vector>
#include <cstddef> // for size_t
#include <numeric> // if needed

namespace dl {
namespace utils {

inline std::vector<size_t> computeStrides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

inline std::vector<size_t> unravelIndex(size_t idx, const std::vector<size_t>& shape) {
    std::vector<size_t> coords(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t s = 1;
        for (size_t j = i + 1; j < shape.size(); ++j) {
            s *= shape[j];
        }
        coords[i] = (idx / s) % shape[i];
    }
    return coords;
}

inline size_t ravelIndex(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    size_t idx = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        idx += coords[i] * strides[i];
    }
    return idx;
}

template<typename T>
inline T stable_accumulate(const std::vector<T>& values) {
    T sum = 0;
    T c = 0; // Compensation for lost low bits
    for (T v : values) {
        T y = v - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

} // namespace utils
} // namespace dl
