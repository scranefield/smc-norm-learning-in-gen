include("Config.jl")
include("Model.jl")

using Parameters
using Distributions
using Gen

"""
    generate_task()

Generates a random task by sampling a color from a predefined set of colors.

Returns:
- A string representing the randomly selected color from ["red", "blue", "green"]

The function uses categorical distribution with equal probabilities (1/3) for each color.
"""
@gen function generate_task()
    # Define possible colors
    colors = ["red", "blue", "green"]

    # Sample a color index using categorical distribution with equal probabilities
    color = {:color} ~ categorical([1 / 3, 1 / 3, 1 / 3])

    return (colors[color])
end

"""
    zone_probabilities(task, norm)

Calculates probability distribution over zones based on the given task and norm.

Arguments:
- `task`: The color of the current task
- `norm`: A norm object specifying rules (can be No_norm or Norm type)

Returns:
- An array of three probabilities corresponding to each zone

The function:
1. Returns default uniform probabilities [1/3, 1/3, 1/3] if no norm applies
2. Handles obligations and prohibitions differently through specialized handlers
3. Modifies zone probabilities based on the norm's specifications
"""
function zone_probabilities(task, norm)
    color = task

    # Default probabilities when no norm applies
    default_probs = [1 / 3, 1 / 3, 1 / 3]

    # Handle different norm types
    if string(typeof(norm)) == "No_norm"
        return default_probs
    else

        # Route to appropriate handler based on norm type
        if string(typeof(norm)) == "Obl"
            return handle_obligation(norm, color)
        elseif string(typeof(norm)) == "Pro"
            return handle_prohibition(norm, color)
        end
    end
    return default_probs
end

"""
    handle_obligation(obl, task_color)

Processes obligation norms to determine zone probabilities.

Arguments:
- `obl`: Obligation object containing left (color) and right (zone) specifications
- `task_color`: The color of the current task

Returns:
- An array of three probabilities where:
  - If the obligation applies (matching color or "any"), sets probability 1.0 for the specified zone
  - Otherwise returns uniform distribution [1/3, 1/3, 1/3]
"""
function handle_obligation(obl, task_color)
    # Extract color and zone from obligation
    norm_color = obl.left.value
    norm_zone = parse(Int, obl.right.value)

    # If obligation applies to current task color or any color
    if norm_color == task_color || norm_color == "any"
        # Create probability vector with 1.0 at obligated zone
        probs = [0.0, 0.0, 0.0]
        probs[norm_zone] = 1.0
        return probs
    else
        # Return uniform distribution if obligation doesn't apply
        return [1 / 3, 1 / 3, 1 / 3]
    end
end

"""
    handle_prohibition(pro, task_color)

Processes prohibition norms to determine zone probabilities.

Arguments:
- `pro`: Prohibition object containing left (color) and right (zone) specifications
- `task_color`: The color of the current task

Returns:
- An array of three normalized probabilities where:
  - If the prohibition applies, sets probability 0.0 for the prohibited zone
  - Normalizes remaining probabilities to sum to 1.0
  - Returns uniform distribution if prohibition doesn't apply
"""
function handle_prohibition(pro, task_color)
    # Extract color and zone from prohibition
    norm_color = pro.left.value
    norm_zone = parse(Int, pro.right.value)

    # If prohibition applies to current task color or any color
    if norm_color == task_color || norm_color == "any"
        # Set equal probabilities for all zones except prohibited one
        probs = [0.5, 0.5, 0.5]
        probs[norm_zone] = 0.0
        return probs ./ sum(probs)
    else
        # Return uniform distribution if prohibition doesn't apply
        return [1 / 3, 1 / 3, 1 / 3]
    end
end

"""
    task_norm_zone(config::NormConfig)

Generates a complete task-norm-zone configuration.

Arguments:
- `config`: Configuration object containing norm generation parameters

Returns:
- A tuple (task, norm, zone) where:
  - task: Generated color task
  - norm: Generated norm structure
  - zone: Selected zone based on task and norm probabilities

Combines task generation, norm generation, and zone selection into a single process.
"""
@gen function task_norm_zone(config::NormConfig)
    # Generate random task (color)
    task = {:task} ~ generate_task()

    # Generate norm structure
    norm = {:norm} ~ prior(1, "NORMS", config)

    # Calculate zone probabilities based on task and norm
    zone_probs = zone_probabilities(task, norm)

    # Sample zone based on calculated probabilities
    zone = {:zone} ~ categorical(zone_probs)
    return (task, norm, zone)
end


"""
    generateNorm(n, constraints)

Generates multiple valid norm configurations based on given constraints.

Arguments:
- `n`: Number of norm configurations to generate
- `constraints`: Constraints to apply during generation

Returns:
- Tuple of (NormSet, WeightSet) where:
  - NormSet: Array of valid generated norms
  - WeightSet: Array of corresponding weights (exp of log probabilities)

Filters out invalid configurations (weight = -Inf) during generation.
"""
function generateNorm(n, constraints)
    config = NormConfig()
    NormSet = []
    WeightSet = []
    for i in 1:n
        # Generate trace and weight
        (trace, weight) = generate(task_norm_zone, (config,), constraints)
        
        # Only keep valid configurations (finite weight)
        if weight > -Inf
            push!(NormSet, Gen.get_retval(trace)[2])
            push!(WeightSet, exp(weight))
        end
    end
    return NormSet, WeightSet
end