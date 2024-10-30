include("Config.jl")
include("Model.jl")
include("SRTree.jl")
include("Task.jl")

# include("inference_rejuv_tree_sr.jl")

config = NormConfig()
timepoints = Vector{Float64}(1:10)
trace1 = Gen.simulate(model, (config,))

proposal_fwd_trace = Gen.simulate(subtree_replace_proposal, (trace1, config))

proposal_retval = Gen.get_retval(proposal_fwd_trace)
proposal_choices = Gen.get_choices(proposal_fwd_trace)

# proposal_fwd_weight = Gen.get_score(proposal_fwd_trace)
println("\n\n------------------\n")

model_trace_out, proposal_choices_out, weight = subtree_replace_involution(trace1, proposal_choices, proposal_retval, (trace1, false))

println("----- model trace ---")
Base.show(stdout, "text/plain", get_choices(trace1))
println("-------------")

println("---- model ---")
println(Gen.get_retval(trace1))
println("-------------")

println("----- proposal trace ---")
Base.show(stdout, "text/plain", get_choices(proposal_fwd_trace))
println("-------------")

println("---- proposal retval ----")
println(get_retval(proposal_fwd_trace))
println("-------------")

println("----- new model trace ---")
Base.show(stdout, "text/plain", get_choices(model_trace_out))
println("-------------")

println("---- new model ----")
println(get_retval(model_trace_out))
println("-------------")

# If you remove Gen.get_retval() from line 20 and [] from line 29, you can see the actual trace.