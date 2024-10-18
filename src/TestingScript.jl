include("Config.jl")
include("Model.jl")
include("SRTree.jl")
include("Task.jl")

# --------------------------------- Norm inference ---------------------------------
config = NormConfig()
trace1 = Gen.simulate(model, (config,))

# Prior norm tree
println("\n\n----- Orignal -----")
println(Gen.get_retval(trace1))
println("-------------------\n")

proposal_fwd_trace = Gen.simulate(subtree_replace_proposal, (trace1, config))
proposal_retval = Gen.get_retval(proposal_fwd_trace)
proposal_choices = Gen.get_choices(proposal_fwd_trace)

# Proposal of subtree
println("----- Proposal -----")
println(proposal_retval)
println("--------------------\n")

model_trace_out, proposal_choices_out, weight = subtree_replace_involution(trace1, proposal_choices, proposal_retval, (trace1, false))

# Replaced tree
println("----- Involution -----")
println(model_trace_out[])
println("----------------------\n")


# --------------------------------- Norm based on observation ---------------------------------
constraints = choicemap((:color, 3), (:zone, 1))
normSet, weightSet = generateNorm(10, constraints)
println("----- Observation -----")
println(constraints)
println("-----------------------\n")

println("----- Possible Norms -----")
println(normSet)
println("--------------------------\n")

println("----- Associated Weights -----")
println(weightSet)
println("------------------------------\n")