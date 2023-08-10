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
            return symbolic_trace(prob, sys)

        end
    catch e
        error_analysis(sys, e)
    end
end

function symbolic_trace(prob, sys)
    # u = get_states(sys);
    # u1inds = findall(x->?, u)
    # u2inds = findall(x->?, u)
    # u1 = u[u1inds]
    # u2 = u[u2inds]
    # du1 = prob.f([collect(u1); collect(u2)], nothing, 0.0)[u1inds];
    # du2 = prob.f([collect(u1); collect(u2)], nothing, 0.0)[u2inds];
    # gen_drho = eval(Symbolics.build_function(du1, collect(u1), collect(u2))[2]);
    # gen_dphi = eval(Symbolics.build_function(du2, collect(u1), collect(u2))[2]);
    # dynamical_f1(_du1,u,p,t) = gen_drho(_du1, u[u1inds], u[u2inds]);
    # dynamical_f2(_du2,u,p,t) = gen_dphi(_du2, u[u1inds], u[u2inds]);
    # u0 = [prob.u0[u1inds]; prob.u0[u2inds]];
    # return DynamicalODEProblem{false}(dynamical_f1, dynamical_f2, u0[1:length(u1)], u0[length(u1)+1:end], prob.tspan)
    len = floor(Int, length(prob.u0)/2);

    @variables rho[1:len] phi[1:len+1]
    drho = (prob.f([collect(rho); collect(phi)], nothing, 0.0)[1:len]);
    dphi = (prob.f([collect(rho); collect(phi)], nothing, 0.0)[len+1:end]);

    gen_drho = eval(Symbolics.build_function(drho, collect(rho), collect(phi))[2]);
    gen_dphi = eval(Symbolics.build_function(dphi, collect(rho), collect(phi))[2]);

    dynamical_f1(_drho,u,p,t) = gen_drho(_drho, u[1:len], u[len+1:end]);
    dynamical_f2(_dphi,u,p,t) = gen_dphi(_dphi, u[1:len], u[len+1:end]);
    u0 = [prob.u0[1:len]; prob.u0[len+1:end]];
    return DynamicalODEProblem{false}(dynamical_f1, dynamical_f2, u0[1:len], u0[len+1:end], prob.tspan)
end
