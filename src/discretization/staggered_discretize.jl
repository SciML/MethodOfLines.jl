function SciMLBase.discretize(
        pdesys::PDESystem,
        discretization::MOLFiniteDifference{G},
        analytic = nothing, kwargs...
    ) where {G <: StaggeredGrid}
    reject_point_ports(pdesys, "discretize")
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
            return staggered_split(prob, simpsys, edge_aligned_dims(discretization))
        end
    catch e
        error_analysis(sys, e)
    end
end

# `f1` holds the time derivatives of the variables edge aligned in an odd number of
# directions and `f2` those of every other variable. A staggered derivative moves its
# variable half a step in one direction, so it always couples the two groups, while
# pointwise terms stay within one: for Maxwell's equations the groups are E and H.
# Each half evaluates the full right hand side and zeroes the other group's rows, so
# `f1 + f2` is exactly `f` for any number of variables, with time, parameters and
# boundary values live, at the price of a second evaluation of `f` per step.
function staggered_split(prob, sys, edge_dims)
    get_var_from_state(state) = operation(arguments(state)[1])
    nedge = Dict(nameof(op) => length(dims) for (op, dims) in edge_dims)
    isedge = [
        isodd(get(nedge, nameof(get_var_from_state(u)), 0)) for u in get_unknowns(sys)
    ]
    rows1, rows2 = findall(isedge), findall(!, isedge)
    f1(du, u, p, t) = (prob.f(du, u, p, t); du[rows2] .= 0; nothing)
    f2(du, u, p, t) = (prob.f(du, u, p, t); du[rows1] .= 0; nothing)
    return SplitODEProblem(f1, f2, prob.u0, prob.tspan, prob.p)
end
