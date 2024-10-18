using Parameters
using Distributions
using Gen

"""
    transform_param(idx::Int, rule::String, param::String, config::NormConfig)

Transform a parameter into a concrete value based on probabilistic rules.

Arguments:
- `idx::Int`: Index identifying the current node in the tree
- `rule::String`: The current rule being applied (e.g., "NORMS", "ZONE", "COLOUR")
- `param::String`: Parameter to transform (e.g., "Colour", "Zone", "No_norm")
- `config::NormConfig`: Configuration containing probability distributions

Returns:
- If param exists in config.q_dict: A randomly sampled value according to the
  specified probability distribution
- Otherwise: Returns the original parameter unchanged

This function handles the probabilistic selection of concrete values for
terminal nodes in the normative tree. For example, when param is "Colour",
it samples from the distribution over specific colors defined in config.
"""
@gen function transform_param(idx::Int, rule::String, param::String, config::NormConfig)
    # Check if we have a probability distribution for this parameter
    if haskey(config.q_dict, param)
        # Extract possible values and their probabilities
        options = collect(keys(config.q_dict[param]))
        probs = collect(values(config.q_dict[param]))

        # Sample a value according to the probability distribution
        value_fill = {(idx, Symbol(param))} ~ categorical(probs)

        # Return the selected option
        return options[value_fill]
    else
        error("No distribution found")
        # If no distribution is specified, return parameter unchanged
        return param
    end
end

"""
    get_node_dist(rule::String, config::NormConfig)

Retrieve the probability distribution over node types for a given rule.

Arguments:
- `rule::String`: The rule to look up (e.g., "NORMS", "ZONE", "COLOUR")
- `config::NormConfig`: Configuration containing node type distributions

Returns:
- Dictionary mapping node types to their probabilities for the given rule
"""
function get_node_dist(rule::String, config::NormConfig)
    return config.p_dict[rule]
end

"""
    prior(1, "NORMS", config)

Generate a probabilistic tree structure according to the specified rules
and probability distributions.

Arguments:
- `idx::Int`: Index identifying the current node in the tree
- `rule::String`: Current rule being applied (e.g., "NORMS", "ZONE", "COLOUR")
- `config::NormConfig`: Configuration containing rules and distributions

Returns:
- A Node instance representing either:
  1. A leaf node (No_norm, Zone, or Colour) with a sampled value
  2. An internal node (Norm, Obl, or Pro) with recursively generated children

This function implements the generative process for creating normative
structures. It:
1. Samples a node type according to the rule's probability distribution
2. For leaf nodes: Samples a concrete value using transform_param
3. For internal nodes: Recursively generates child nodes according to the rules
4. Enforces the constraint that norm nodes can have at most two children

The resulting tree structure represents a complete normative specification,
combining obligations, prohibitions, and their associated conditions.
"""
@gen function prior(idx::Int, rule::String, config::NormConfig)
    # Get probability distribution over node types for this rule
    node_dist = get_node_dist(rule, config)

    # Sample a node type according to the distribution
    node_type = {(idx, :node_type)} ~ categorical(collect(values(node_dist)))
    node_key = collect(keys(node_dist))[node_type]

    # Check if this is a terminal rule (leads to a leaf node)
    if isempty(config.rule_dict[rule][node_key])
        # Create a leaf node
        struct_name = Symbol(node_key)
        constructor = eval(struct_name)

        # Sample a concrete value for the leaf node
        value_sampled = {*} ~ transform_param(idx, rule, node_key, config)
        return constructor(value_sampled)
    else
        # Create an internal node
        # Calculate indices for child nodes
        idx_l = Gen.get_child(idx, 1, 2)
        idx_r = Gen.get_child(idx, 2, 2)

        # Get the rules that should be applied to child nodes
        child_rules = config.rule_dict[rule][node_key]

        # Enforce maximum number of children
        if length(child_rules) > 2
            error("Norm nodes can have at most two children")
        end

        # Recursively generate left child
        left_child = {*} ~ prior(idx_l, child_rules[1], config)

        # Handle right child - either EMPTY() or recursively generated
        right_child = EMPTY()
        if length(child_rules) == 2
            right_child = {*} ~ prior(idx_r, child_rules[2], config)
        end

        # Construct and return the internal node
        struct_name = Symbol(node_key)
        constructor = eval(struct_name)
        return constructor(left_child, right_child)
    end
end

"""
    model(config::NormConfig)

Generate a probabilistic model for norm structures.

Arguments:
- `config::NormConfig`: Configuration object containing rules and parameters
  for generating the norm tree

Returns:
- Prior function representing the generated norm tree structure

This is the top-level generative model that creates the initial norm tree
structure using the prior distribution specified in the configuration.
"""
@gen function model(config::NormConfig)
    # Generate initial tree structure
    prior_fn = {:tree} ~ prior(1, "NORMS", config) # generating the tree
    return prior_fn
end