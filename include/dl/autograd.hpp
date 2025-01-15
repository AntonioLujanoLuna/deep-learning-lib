#pragma once

#include <memory>
#include <vector>
#include <iostream>

namespace dl {

class Node {
public:
    virtual ~Node() = default;
    virtual void backward() = 0;
    virtual std::string node_type() const = 0;  // Add this to identify node types
};

class ComputationGraph {
public:
    static ComputationGraph& getInstance() {
        static ComputationGraph instance;
        return instance;
    }

    template<typename T>
    void storeTensor(const T& tensor) {
        tensors_.push_back(std::make_shared<T>(tensor));
    }
    
    template<typename T>
    void storeTensorPtr(const std::shared_ptr<T>& tensor_ptr) {
        tensors_.push_back(tensor_ptr);
    }

    void addNode(std::shared_ptr<Node> node) {
        nodes_.push_back(node);
    }

    void backward() {
        if (nodes_.empty()) {
            std::cout << "No nodes in computation graph, skipping backward pass" << std::endl;
            return;
        }

        // Execute backward pass in reverse order
        size_t node_idx = nodes_.size();
        for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it, --node_idx) {
            try {
                (*it)->backward();
            } catch (const std::exception& e) {
                std::cerr << "Error in backward pass at " << (*it)->node_type() << " node " << node_idx << ": " << e.what() << std::endl;
                throw;
            }
        }
    }

    void clear() {
        nodes_.clear();
        tensors_.clear();
    }

    const std::vector<std::shared_ptr<Node>>& getNodes() const {
        return nodes_;
    }

private:
    ComputationGraph() = default;
    ComputationGraph(const ComputationGraph&) = delete;
    ComputationGraph& operator=(const ComputationGraph&) = delete;
    std::vector<std::shared_ptr<Node>> nodes_;
    std::vector<std::shared_ptr<void>> tensors_;  // Store any type of tensor
};

} // namespace dl
