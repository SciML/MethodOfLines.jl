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
            add_metadata!(getmetadata(simpsys, ModelingToolkit.ProblemTypeCtx, nothing), sys)
            prob = ODEProblem(
                simpsys, Pair[], tspan; discretization.kwargs...,
                kwargs...
            )
            return symbolic_trace(prob, simpsys)
        end
    catch e
        error_analysis(sys, e)
    end
end

function symbolic_trace(prob, sys)
    get_var_from_state(state) = operation(arguments(state)[1])
    unknowns = get_unknowns(sys)
    u1_var = get_var_from_state(unknowns[1])
    u2_var = get_var_from_state(unknowns[findfirst(x -> get_var_from_state(x) != u1_var, unknowns)])
    u1inds = findall(x -> get_var_from_state(x) === u1_var, unknowns)
    u2inds = findall(x -> get_var_from_state(x) === u2_var, unknowns)
    u1 = unknowns[u1inds]
    u2 = unknowns[u2inds]
    tracevec_1 = [i in u2inds ? unknowns[i] : Num(0.0) for i in 1:length(unknowns)] # maybe just 0.0
    tracevec_2 = [i in u1inds ? unknowns[i] : Num(0.0) for i in 1:length(unknowns)]
    du1 = prob.f(tracevec_1, prob.p, 0.0)
    du2 = prob.f(tracevec_2, prob.p, 0.0)
    gen_du1 = eval(Symbolics.build_function(du1, unknowns)[2])
    gen_du2 = eval(Symbolics.build_function(du2, unknowns)[2])
    dynamical_f1(_du1, u, p, t) = gen_du1(_du1, u)
    dynamical_f2(_du2, u, p, t) = gen_du2(_du2, u)
    u0 = prob.u0 #[prob.u0[u1inds]; prob.u0[u2inds]];
    return SplitODEProblem(dynamical_f1, dynamical_f2, u0, prob.tspan)
    #return DynamicalODEProblem{false}(dynamical_f1, dynamical_f2, u0[1:length(u1)], u0[length(u1)+1:end], prob.tspan)
end
