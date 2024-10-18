using Parameters
using Distributions
using Gen

"""
    subtree_replace_proposal(trace, configIn)

Proposes modifications to the norm tree structure through subtree replacement.

Arguments:
- `trace`: Current trace containing the norm tree
- `configIn`: Configuration for norm generation

Returns:
- Tuple (picked_node, picked_idx, picked_depth, subtree_proposed) containing:
  - Information about the selected node to replace
  - The proposed replacement subtree

Implements the proposal step of the MCMC sampling process for norm structures.
"""
@gen function subtree_replace_proposal(trace, configIn)
    # Get current tree structure
    root = get_retval(trace)
    (picked_node, picked_idx, picked_depth) = {:pick_node} ~ pick_random_node(root, 1, 1, false, false)

    # Determine rule type for picked node
    rule = typeof(picked_node) <: LeafNode ? string(typeof(picked_node)) : "NORMS"

    # Generate proposed replacement subtree
    subtree_proposed = {:subtree} ~ prior(picked_idx, uppercase(rule), configIn)
    return (picked_node, picked_idx, picked_depth, subtree_proposed)
end


# Helper function to calculate size of tree nodes
size(n::LeafNode) = 1
size(n::NormNode) = n.size

"""
    get_p_done(node::Node, leaf::Bool, noroot::Bool)

Calculate the probability of terminating the random node selection at the current node.

Arguments:
- `node::Node`: The current node being evaluated
- `leaf::Bool`: If true, only leaf nodes can be selected
- `noroot::Bool`: If true, root node cannot be selected

Returns:
- Float64: Probability of selecting current node, where:
  - For LeafNodes: Returns 1.0 unless noroot is true (error case)
  - For NormNodes: Returns 0.0 if noroot, 0.0 if leaf-only, else 1/size(node)

The function implements the termination logic for random node selection in the tree,
considering constraints about leaf-only selection and root exclusion.
"""
function get_p_done(node::Node, leaf::Bool, noroot::Bool)
    if node isa LeafNode

        # Can't pick leaf node if noroot is true
        return noroot ? error("Impossible pick_random_node call.") : 1.0
    elseif node isa NormNode

        # Probability calculations for norm nodes
        return noroot ? 0.0 : leaf ? 0.0 : 1.0 / size(node)
    end
end

"""
    get_p_recurse_left(node::NormNode)

Calculate the probability of recursing to the left child during random node selection.

Arguments:
- `node::NormNode`: The current internal node in the tree

Returns:
- Float64: Probability of choosing left child, calculated as:
  (size of left subtree) / (total size - 1)

This function ensures that the probability of selecting any node in the tree
is proportional to the size of the subtree rooted at that node.
"""
function get_p_recurse_left(node::NormNode)

    # Probability proportional to size of left subtree
    return size(node.left) / (size(node) - 1)
end


"""
    pick_random_node(node::Node, idx::Int, depth::Int, leaf::Bool, noroot::Bool)

Recursively selects a random node from a norm tree structure.

Arguments:
- `node`: Current node in the tree
- `idx`: Index of current node
- `depth`: Depth of current node in tree
- `leaf`: Whether to only select leaf nodes
- `noroot`: Whether to exclude root node from selection

Returns:
- Tuple (selected_node, index, depth) of the randomly chosen node

Implements a probabilistic tree traversal algorithm that:
1. Can terminate at current node with probability based on node type
2. Otherwise recurses to children with probabilities proportional to subtree sizes
"""
@gen function pick_random_node(
    node::Node,  # current node
    idx::Int,    # index of the current node
    depth::Int,  # depth of the current node
    leaf::Bool,  # leaves only
    noroot::Bool # disallow selecting root
)
    # Check if we should terminate at current node
    p_done = get_p_done(node, leaf, noroot)
    if ({:done => depth} ~ bernoulli(p_done))
        return (node, idx, depth)
    end

    # Handle norm nodes (internal nodes)
    if node isa NormNode
        # Calculate probability of going to left child
        p_recurse_left = get_p_recurse_left(node)

        # Either recurse to left or right child
        if ({:recurse_left => idx} ~ bernoulli(p_recurse_left))
            return {*} ~ pick_random_node(node.left, 2 * idx, depth + 1, leaf, false)
            # Recurse to right child
        else
            return {*} ~ pick_random_node(node.right, 2 * idx + 1, depth + 1, leaf, false)
        end
    else
        error("Unexpected node type")
    end
end



"""
    subtree_replace_involution(model_trace, proposal_choices_in, proposal_retval, proposal_args)

Implement the involution (reversible operation) for the MCMC subtree replacement proposal.

Arguments:
- `model_trace::Gen.Trace`: Current model trace
- `proposal_choices_in::Gen.ChoiceMap`: Proposed changes
- `proposal_retval`: Return value from proposal (contains subtree information)
- `proposal_args::Tuple`: Additional arguments for the proposal

Returns:
- Tuple (model_trace_out, weight): Updated trace and corresponding weight

This function implements the reversible jump MCMC update step for modifying
the norm tree structure, ensuring that the Markov chain maintains detailed balance.
"""
function subtree_replace_involution(
    model_trace::Gen.Trace,
    proposal_choices_in::Gen.ChoiceMap,
    proposal_retval,
    proposal_args::Tuple)

    # Extract proposal information
    (subtree_idx, subtree_depth, subtree_proposed) = proposal_retval

    # Create new model trace with proposed changes
    model_choices_diff = choicemap()
    proposal_choices_subtree = get_submap(proposal_choices_in, :subtree)

    # Update model with proposed subtree
    set_submap!(model_choices_diff, :tree, proposal_choices_subtree)
    (model_trace_out, weight, retdiff, discard) = update(model_trace, model_choices_diff)

    # Prepare reverse proposal choices
    proposal_choices_out = choicemap()
    proposal_choices_pick_node = get_submap(proposal_choices_in, :pick_node)
    set_submap!(proposal_choices_out, :pick_node, proposal_choices_pick_node)
    set_submap!(proposal_choices_out, :subtree, get_submap(discard, :tree))

    return (model_trace_out, proposal_choices_out, weight)

end