# C++ Neural Network Library with Autograd

A modern C++ neural network library featuring automatic differentiation and architecture-agnostic design.

## Features

- Automatic differentiation (autograd) engine
- Architecture-agnostic design (CPU, GPU support)
- Modern C++ implementation
- Modular neural network layers
- Basic optimization algorithms

## Building the Project

### Prerequisites

- CMake (>= 3.15)
- C++17 compatible compiler
- Git

### Build Instructions

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Running Tests

```bash
cd build
ctest
```

## Project Structure

```
.
├── CMakeLists.txt
├── README.md
├── include/
│   └── dl/
│       ├── tensor.hpp
│       ├── autograd.hpp
│       └── ops/
├── src/
│   └── CMakeLists.txt
└── tests/
    └── CMakeLists.txt
```

## License

MIT License
