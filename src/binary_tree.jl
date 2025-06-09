"""
    Node{T}

Binary tree node used internally by the package.

# Fields
- `val::T` : value stored in the node.
- `idx::BitArray` : binary representation of the node position in the tree.
- `left::Node{T}` : reference to the left child node.
- `right::Node{T}` : reference to the right child node.
"""
mutable struct Node{T}
    val::T
    idx::BitArray
    left::Node{T}
    right::Node{T}
end