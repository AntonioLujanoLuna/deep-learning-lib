// File: C:\Users\aluja\Desktop\DL\include\dl\autograd.hpp
#pragma once

#include "dl/fwd.hpp"
#include <memory>
#include <unordered_set>
#include <functional>
#include <vector>
#include <iostream>

namespace dl {

class Node : public std::enable_shared_from_this<Node> {
public:
    virtual ~Node() = default;
    virtual void backward() = 0;
    virtual std::string node_type() const = 0;  
    
    // DAG structure
    std::vector<std::weak_ptr<Node>> parents_;
    std::vector<std::weak_ptr<Node>> children_;
};

class ComputationGraph {
public:
    // Singleton accessor
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

    /**
     * Backward pass using DAG-based topological order.
     * @param final_node The final node (usually the loss node) from which to start the backward pass.
     */
    void backward(std::shared_ptr<Node> final_node) {
        if (!final_node) {
            std::cerr << "[Warning] Final node is null, skipping backward.\n";
            return;
        }

        // Set to track visited nodes to prevent revisiting
        std::unordered_set<Node*> visited;
        // Vector to store nodes in topological order
        std::vector<std::shared_ptr<Node>> topo_order;

        // Recursive DFS lambda to perform topological sorting
        std::function<void(std::shared_ptr<Node>)> dfs = [&](std::shared_ptr<Node> current) {
            if (!current) return;
            if (visited.find(current.get()) != visited.end()) {
                return;  // Node already visited
            }
            visited.insert(current.get());

            // Recursively visit all parent nodes
            for (auto &weakParent : current->parents_) {
                auto parent = weakParent.lock();
                if (parent) {
                    dfs(parent);
                }
            }

            // After visiting parents, add the current node to topo_order
            topo_order.push_back(current);
        };

        // Initiate DFS from the final_node
        dfs(final_node);

        // Perform backward pass in topological order
        for (auto &node : topo_order) {
            node->backward();
        }
    }

    /**
     * Optional: Existing no-argument backward method using reverse iteration.
     * This can be kept for backward compatibility or removed if you prefer solely using the DAG-based approach.
     */
    void backward() {
        // Reverse iterate over all nodes and call backward
        for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
            (*it)->backward();
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
