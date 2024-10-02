mutable struct Node{T}
    val::T
    idx::BitArray
    left::Node{T}
    right::Node{T}
end