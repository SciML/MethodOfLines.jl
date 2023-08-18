function SciMLBase.discretize(pdesys::PDESystem,
                              discretization::MOLFiniteDifference{G},
                              analytic = nothing, kwargs...) where {G<:StaggeredGrid}
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    try
        simpsys = structural_simplify(sys)
        if tspan === nothing
            add_metadata!(get_metadata(sys), sys)
            return prob = NonlinearProblem(simpsys, ones(length(simpsys.states));
                                           discretization.kwargs..., kwargs...)
        else
            add_metadata!(get_metadata(simpsys), sys)
            prob = ODEProblem(simpsys, Pair[], tspan; discretization.kwargs...,
                              kwargs...)
            return symbolic_trace(prob, simpsys)

        end
    catch e
        error_analysis(sys, e)
    end
end

function symbolic_trace(prob, sys)
    get_var_from_state(state) = operation(arguments(state)[1]);
    states = get_states(sys);
    u1_var = get_var_from_state(states[1]);
    u2_var = get_var_from_state(states[findfirst(x->get_var_from_state(x)!=u1_var, states)]);
    u1inds = findall(x->get_var_from_state(x)===u1_var, states);
    u2inds = findall(x->get_var_from_state(x)===u2_var, states);
    u1 = states[u1inds]
    u2 = states[u2inds]
    du1 = prob.f([collect(u1); collect(u2)], nothing, 0.0)[u1inds];
    du2 = prob.f([collect(u1); collect(u2)], nothing, 0.0)[u2inds];
    gen_du1 = eval(Symbolics.build_function(du1, collect(u1), collect(u2))[2]);
    gen_du2 = eval(Symbolics.build_function(du2, collect(u1), collect(u2))[2]);
    dynamical_f1(_du1,u,p,t) = gen_du1(_du1, u[u1inds], u[u2inds]);
    dynamical_f2(_du2,u,p,t) = gen_du2(_du2, u[u1inds], u[u2inds]);
    u0 = [prob.u0[u1inds]; prob.u0[u2inds]];
    return DynamicalODEProblem{false}(dynamical_f1, dynamical_f2, u0[1:length(u1)], u0[length(u1)+1:end], prob.tspan)
end
