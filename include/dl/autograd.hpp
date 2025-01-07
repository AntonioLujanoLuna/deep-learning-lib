#pragma once

#include <memory>
#include <vector>
#include <iostream>

namespace dl {

class Node {
public:
    virtual ~Node() = default;
    virtual void backward() = 0;
};

class ComputationGraph {
public:
    static ComputationGraph& getInstance() {
        static ComputationGraph instance;
        return instance;
    }

    void addNode(std::shared_ptr<Node> node) {
        nodes_.push_back(node);
    }

    void backward() {
        if (nodes_.empty()) {
            std::cout << "Warning: No nodes in computation graph during backward pass" << std::endl;
            return;
        }

        std::cout << "Starting backward pass with " << nodes_.size() << " nodes" << std::endl;
        
        // Execute backward pass in reverse order
        for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
            try {
                (*it)->backward();
            } catch (const std::exception& e) {
                std::cout << "Error in backward pass: " << e.what() << std::endl;
                throw;
            }
        }
        
        std::cout << "Finished backward pass" << std::endl;
    }

    void clear() {
        if (!nodes_.empty()) {
            std::cout << "Clearing computation graph with " << nodes_.size() << " nodes" << std::endl;
            nodes_.clear();
        }
    }

    const std::vector<std::shared_ptr<Node>>& getNodes() const {
        return nodes_;
    }

private:
    ComputationGraph() = default;
    ComputationGraph(const ComputationGraph&) = delete;
    ComputationGraph& operator=(const ComputationGraph&) = delete;
    std::vector<std::shared_ptr<Node>> nodes_;
};

} // namespace dl
