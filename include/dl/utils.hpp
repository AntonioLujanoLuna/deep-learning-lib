#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace dl {
namespace utils {

inline std::string shape_to_string(const std::vector<size_t>& shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

} // namespace utils
} // namespace dl
