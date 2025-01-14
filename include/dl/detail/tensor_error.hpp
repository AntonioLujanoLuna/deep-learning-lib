#pragma once

#include <stdexcept>
#include <string>

namespace dl {
namespace detail {

class TensorError : public std::runtime_error {
public:
    explicit TensorError(const std::string& message) 
        : std::runtime_error(message) {}
};

} // namespace detail
} // namespace dl
