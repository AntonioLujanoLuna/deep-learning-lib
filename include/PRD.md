Below is a **Product Requirements Document (PRD)** listing out the **exact steps** needed to reorganize and improve your repository for branching autodiff and a cleaner header-only structure. You can feed this document into another LLM or follow it manually.

---

# PRD: Repository Refactoring & Autograd Enhancements

## **1. Introduce Forward Declarations**

1. **Create a new header**:  
   - Name: `include/dl/fwd.hpp`  
   - Purpose: Declare forward references to classes that appear in multiple headers (e.g., `Node`, `Tensor`, etc.) without including their full definitions.  
   - Example content:
     ```cpp
     #pragma once

     namespace dl {
         class Node; 
         // Optionally, forward-declare TensorImpl, Tensor, etc.
     }
     ```

2. **Remove direct includes** of `autograd.hpp` or other heavy headers from files that only need a pointer/reference to `Node`.  
   - Instead, `#include "dl/fwd.hpp"`, then use `std::weak_ptr<dl::Node>` or `std::shared_ptr<dl::Node>`.

**Goal**: Avoid circular dependencies and reduce compile-time overhead.

---

## **2. Update `TensorImpl` and `Tensor` to Store a `grad_fn` Pointer**

1. In `tensor_impl.hpp` (inside `detail`):
   - Include `fwd.hpp` (instead of `autograd.hpp`).  
   - Add a member `std::weak_ptr<dl::Node> grad_fn_;` in `TensorImpl<T>`.

2. In `tensor.hpp`:
   - Include `fwd.hpp` and `tensor_impl.hpp`.  
   - Add `void setGradFn(const std::weak_ptr<Node>& node);` and `std::weak_ptr<Node> gradFn() const;` to `class Tensor`.  
   - In `setGradFn`, set `impl_->grad_fn_ = node;`  
   - In `gradFn`, return `impl_->grad_fn_;`

**Goal**: Let each `Tensor` reference the `Node` that produced it, enabling a DAG for backprop.

---

## **3. Overhaul the `Node` Class for Branching**

1. In `autograd.hpp`, define `class Node` fully:
   - Include `fwd.hpp` as needed.  
   - Derive from `std::enable_shared_from_this<Node>`.  
   - Add containers for:
     ```cpp
     std::vector<std::weak_ptr<Node>> parents_;
     std::vector<std::weak_ptr<Node>> children_;
     ```
   - Provide a pure virtual `void backward()` method and a `std::string node_type() const` method.

2. **(Optional)**: If you have special data structures for partial forward references, consider them here. Otherwise, store minimal references and let operator nodes handle the rest.

**Goal**: A Node can see which Nodes produce its inputs (`parents_`) and which Nodes consume its output (`children_`).

---

## **4. Enable Topological Backward in `ComputationGraph`**

1. Remove or deprecate the current “reverse `std::vector`” backward pass.  
2. Add a method `void backward(std::shared_ptr<Node> final_node)`:
   - Perform a DFS-based or Kahn-based topological sort starting at `final_node`.  
   - Collect all reachable parents in correct order.  
   - Call `node->backward()` in the resulting topological order.

3. Ensure you provide a way to call `backward()` from the final node (e.g., the node that produced the loss):
   - `auto loss_node = loss.gradFn().lock();`
   - `ComputationGraph::getInstance().backward(loss_node);`

4. (Optional) Decide whether to `clear()` the graph after backward or not.

**Goal**: Properly handle branching graphs without relying on a single linear chain of nodes.

---

## **5. Modify Each Operator Node to Link Parents and Children**

1. **In operator constructors** (e.g., `AddNode`, `MulNode`, `MatMulNode`, etc.):
   - For each input Tensor:
     1. `auto parent = input.gradFn().lock();`
     2. If `parent` is non-null, do:
        ```cpp
        parents_.push_back(parent);
        parent->children_.push_back(shared_from_this());
        ```
   - For the output Tensor, do `output.setGradFn(shared_from_this());`
   - Register the node with `ComputationGraph::getInstance().addNode(shared_from_this());`
   - (Optionally) perform the forward pass in the constructor if it’s truly “eager.”

2. **Implement `backward()`** in each operator node:
   - Retrieve `out_grad` (the gradient w.r.t. node’s output).  
   - Compute partial derivatives for each parent input.  
   - Accumulate those into the parents’ Tensor gradients (e.g., `a.grad()[i] += out_grad[i] * ...;`).

**Goal**: Build a DAG of Node objects that accurately represents the computation, so gradient flow is correct.

---

## **6. Validate/Refine Gradient Accumulation in `Tensor`**

1. Ensure `Tensor<T>` (and internally `TensorImpl<T>`) has a vector for gradients (`std::vector<T> grad_`) that:
   - Is allocated the same size as `data_`.  
   - Can be zeroed out by `zero_grad()`.

2. Check that each node’s `backward()` calls are **adding** (`+=`) to `grad_` for each relevant parent.

3. (Optional) Provide user-friendly methods:
   - `void Tensor<T>::zero_grad();`
   - `void Tensor<T>::accumulate_grad(std::vector<T> delta);`

**Goal**: Each parent’s gradient is the **sum** of contributions from multiple children.

---

## **7. Restructure for Header-Only with Clear Separation**

1. Keep all public headers in `include/dl/...`.  
2. **Use `detail/`** for purely internal or low-level classes (`tensor_impl.hpp`, `tensor_error.hpp`).  
3. **Use** a `fwd.hpp` to break cycles.  
4. **Keep** each subcomponent (`ops`, `nn`, `optim`, `utils`) in its own subfolder.  
5. (Optional) Provide a **single** top-level `dl.hpp` that includes everything, making it simpler for users to do:
   ```cpp
   #include <dl/dl.hpp>
   ```

**Goal**: Provide a structured, minimal-dependency set of headers that is easy to navigate and maintain.

---

## **8. Unit Tests & Gradient Checking**

1. **Extend** your existing tests to create branching graphs:
   - Example: `z = x + y; w1 = x * z; w2 = y * z; loss = w1 + w2;`
   - Check final gradients for correctness.  
2. **Add** gradient checking:
   - Numerically approximate partial derivatives for small-scale ops.  
   - Compare with symbolic results.

**Goal**: Confirm that your new DAG-based system accumulates gradients correctly for multi-branch scenarios.

---

## **9. Documentation & Comments**

1. Use **Doxygen** or another doc-generator format to annotate classes/functions:
   - Document the relationship between `Tensor` and `Node`.
   - Clarify that each operator node is responsible for linking inputs/outputs in the graph.

2. Provide **inline comments** explaining the topological traversal logic.  
3. Maintain a short **README** or `doc/` directory explaining how to build and use the library.

**Goal**: Ensure future contributors (and yourself) can navigate the code easily.

---

## **10. (Optional) Additional Enhancements**

- **Parallelization**: Add parallel loops (e.g., OpenMP) in large ops if needed.  
- **GPU Support**: Abstract device logic if you plan to extend to CUDA/OpenCL in the future.  
- **Optimizer Overhaul**: Add advanced optimizers (Adam, RMSProp) that properly handle the updated gradient logic.  
- **Continuous Integration**: Configure a CI service (e.g., GitHub Actions) to build/test on each commit.

**Goal**: Keep the door open for performance and usability improvements once the DAG-based design is stable.

---

## **Summary of Required Steps**

1. **Create** `fwd.hpp` with forward declarations (`Node`, etc.).  
2. **Modify** `tensor_impl.hpp` & `tensor.hpp` to store a `std::weak_ptr<Node> grad_fn_`.  
3. **Revise** `Node` in `autograd.hpp` to have `parents_`, `children_`, and a `backward()` method.  
4. **Implement** a topological backward in `ComputationGraph::backward(std::shared_ptr<Node>)`.  
5. **Update** each operator node constructor (`AddNode`, `MulNode`, etc.) to link parents/children properly.  
6. **Ensure** gradient accumulation is correct in each node’s `backward()`.  
7. **Refine** your header-only structure with `detail/`, `ops/`, `nn/`, `optim/`, `utils/`, and possibly `dl.hpp`.  
8. **Expand** tests for branching autodiff and do gradient-checking.  
9. **Document** everything with comments/doxygen.  
10. **(Optional)** Add concurrency, GPU, advanced optimizers, or other features as needed.

---

## **End of PRD**

Follow these steps precisely to reorganize and enhance your repository. This will give you a **clean** header-only library supporting **branching autodiff** with minimal circular dependencies.