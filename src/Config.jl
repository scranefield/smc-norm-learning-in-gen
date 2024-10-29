using Parameters
using Distributions
using Gen

"""
Basic Type Hierarchy:
- Node: Abstract base type for all nodes
  |- LeafNode: For terminal nodes (no children)
  |- NormNode: For nodes that can have children
"""
abstract type Node end
abstract type LeafNode <: Node end
abstract type NormNode <: Node end

struct No_norm <: LeafNode
    value::String
end

struct Zone <: LeafNode
    value::String
end

struct Colour <: LeafNode
    value::String
end

struct EMPTY <: LeafNode
end

"""
    Norm <: NormNode

Node representing a general normative structure combining other nodes.
Fields:
- `left::Node`: Left child node (typically contains the main norm)
- `right::Node`: Right child node (typically contains conditions)
- `size::Int`: Total number of nodes in the subtree rooted here
- `depth::Int`: Maximum depth of the subtree rooted here
"""
struct Norm <: NormNode
    left::Node
    right::Node
    size::Int
    depth::Int
end

"""
    Norm(left, right)

Constructor for Norm that automatically calculates size and depth.
"""
function Norm(left, right)
    s = 1 + size(left) + size(right)
    d = 1 + max(depth(left), depth(right))
    return Norm(left, right, s, d)
end

"""
    Obl <: NormNode

Node representing an obligation ("must do") norm.
Fields:
- `left::Node`: Left child specifying what must be done (typically Colour)
- `right::Node`: Right child specifying where it applies (typically Zone)
- `size::Int`: Total number of nodes in the subtree
- `depth::Int`: Maximum depth of the subtree
"""
struct Obl <: NormNode
    left::Node
    right::Node
    size::Int
    depth::Int
end

"""
    Obl(left, right)

Constructor for Obl that automatically calculates size and depth.
"""
function Obl(left, right)
    s = 1 + size(left) + size(right)
    d = 1 + max(depth(left), depth(right))
    return Obl(left, right, s, d)
end

"""
    Pro <: NormNode

Node representing a prohibition ("must not do") norm.
Fields:
- `left::Node`: Left child specifying what is prohibited (typically Colour)
- `right::Node`: Right child specifying where it applies (typically Zone)
- `size::Int`: Total number of nodes in the subtree
- `depth::Int`: Maximum depth of the subtree
"""
struct Pro <: NormNode
    left::Node
    right::Node
    size::Int
    depth::Int
end

"""
    Pro(left, right)

Constructor for Pro that automatically calculates size and depth.
"""
function Pro(left, right)
    s = 1 + size(left) + size(right)
    d = 1 + max(depth(left), depth(right))
    return Pro(left, right, s, d)
end

# Define size and depth for LeafNodes
size(n::LeafNode) = 1
depth(n::LeafNode) = 0

# Define size and depth for NormNodes
size(n::NormNode) = n.size
depth(n::NormNode) = n.depth

"""
    config = NormConfig()

Configuration structure for the normative reasoning system.
Fields:
- `rule_dict::Dict`: Maps node types to their allowed child node types
- `p_dict::Dict`: Probability distributions for selecting node types
- `q_dict::Dict`: Probability distributions for terminal values

The configuration defines:
1. Valid rule expansions (what children each node type can have)
2. Probabilities for selecting different node types
3. Probabilities for selecting specific terminal values
"""
@with_kw struct NormConfig
    # Rule dictionary
    rule_dict::Dict{String,Dict{String,Vector{String}}} = Dict(
        "NORMS" => Dict("No_norm" => [], "Obl" => ["COLOUR", "ZONE"], "Pro" => ["COLOUR", "ZONE"]),
        "ZONE" => Dict("Zone" => []),
        "COLOUR" => Dict("Colour" => []),
        "NO_NORM" => Dict("No_norm" => [])
    )
    # Probability dictionaries
    p_dict::Dict{String,Dict{String,Float64}} = Dict(
        "NORMS" => Dict("No_norm" => 0.3, "Obl" => 0.4, "Pro" => 0.3),
        "ZONE" => Dict("Zone" => 1.0),
        "COLOUR" => Dict("Colour" => 1.0),
        "NO_NORM" => Dict("No_norm" => 1.0)
    )
    # Probability distributions for terminal values
    q_dict::Dict{String,Dict{String,Float64}} = Dict(
        "Colour" => Dict("red" => 1 / 6, "green" => 1 / 2, "blue" => 1 / 6, "any" => 1 / 6),
        "Zone" => Dict("1" => 1 / 2, "2" => 1 / 4, "3" => 1 / 4),
        "No_norm" => Dict("true" => 1.0)
    )
end