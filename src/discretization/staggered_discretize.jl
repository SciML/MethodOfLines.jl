function SciMLBase.discretize(
        pdesys::PDESystem,
        discretization::MOLFiniteDifference{G},
        analytic = nothing, kwargs...
    ) where {G <: StaggeredGrid}
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    return try
        simpsys = mtkcompile(sys)
        if tspan === nothing
            add_metadata!(getmetadata(sys, ModelingToolkit.ProblemTypeCtx, nothing), sys)
            unkns = get_unknowns(simpsys)
            return prob = NonlinearProblem(
                simpsys, unkns .=> ones(length(unkns));
                discretization.kwargs..., kwargs...
            )
        else
            mol_metadata = getmetadata(simpsys, ModelingToolkit.ProblemTypeCtx, nothing)
            add_metadata!(mol_metadata, sys)
            prob = ODEProblem(
                simpsys, nothing, tspan;
                discretization.kwargs...,
                kwargs...
            )
            return symbolic_trace(prob, simpsys, edge_aligned_dims(discretization))
        end
    catch e
        error_analysis(sys, e)
    end
end

# `f1` holds the time derivatives of the variables edge aligned in an odd number of
# directions and `f2` those of every other variable. A staggered derivative moves its
# variable half a step in one direction, so it always couples the two groups, while
# pointwise terms stay within one: for Maxwell's equations the groups are E and H.
# Splitting the rows of `du`, rather than the states `du` is evaluated at, keeps
# `f1 + f2` equal to the full right hand side for any number of variables.
function symbolic_trace(prob, sys, edge_dims)
    get_var_from_state(state) = operation(arguments(state)[1])
    unknowns = get_unknowns(sys)
    nedge = Dict(nameof(op) => length(dims) for (op, dims) in edge_dims)
    isedge = [isodd(get(nedge, nameof(get_var_from_state(u)), 0)) for u in unknowns]
    # traced at a symbolic time so time dependent boundary values and sources stay live
    iv = ModelingToolkit.get_iv(sys)
    du = prob.f(unknowns, prob.p, iv)
    du1 = [isedge[i] ? du[i] : Num(0.0) for i in eachindex(du)]
    du2 = [isedge[i] ? Num(0.0) : du[i] for i in eachindex(du)]
    gen_du1 = Symbolics.build_function(du1, unknowns, iv; expression = Val{false})[2]
    gen_du2 = Symbolics.build_function(du2, unknowns, iv; expression = Val{false})[2]
    dynamical_f1(_du1, u, p, t) = gen_du1(_du1, u, t)
    dynamical_f2(_du2, u, p, t) = gen_du2(_du2, u, t)
    u0 = prob.u0
    return SplitODEProblem(dynamical_f1, dynamical_f2, u0, prob.tspan)
    #return DynamicalODEProblem{false}(dynamical_f1, dynamical_f2, u0[1:length(u1)], u0[length(u1)+1:end], prob.tspan)
end
