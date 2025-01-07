#pragma once

#include <memory>
#include <vector>

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
        // Traverse nodes in reverse order for backward pass
        for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
            if (*it) {
                (*it)->backward();
            }
        }
    }

    void clear() {
        nodes_.clear();
    }

private:
    ComputationGraph() = default;
    ComputationGraph(const ComputationGraph&) = delete;
    ComputationGraph& operator=(const ComputationGraph&) = delete;
    std::vector<std::shared_ptr<Node>> nodes_;
};

} // namespace dl
