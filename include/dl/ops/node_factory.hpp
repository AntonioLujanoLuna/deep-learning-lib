// node_factory.hpp
#pragma once
#include <memory>
#include "node.hpp"
#include "autograd.hpp" // Assuming this includes ComputationGraph
#include "tensor.hpp"

namespace dl {
    namespace ops {

        // Generic factory function
        template<typename NodeType, typename... Args>
        std::shared_ptr<NodeType> createNode(Args&&... args) {
            // Create the node instance
            auto node = std::make_shared<NodeType>(std::forward<Args>(args)...);

            // 1) Set the gradFn for the output tensor
            // Assuming the last argument is the output tensor
            // Adjust the index if necessary
            auto outputTensor = std::get<1>(std::forward_as_tuple(args...));
            outputTensor->setGradFn(node);

            // 2) Link parents if input tensors have gradFn
            // Assuming the first argument is the input tensor
            // Adjust the index if necessary
            auto inputTensor = std::get<0>(std::forward_as_tuple(args...));
            if (auto parent = inputTensor->gradFn().lock()) {
                node->parents_.push_back(parent);
                parent->children_.push_back(node);
            }

            // 3) Register the node with the ComputationGraph
            ComputationGraph::getInstance().addNode(node);

            return node;
        }

    } // namespace ops
} // namespace dl
