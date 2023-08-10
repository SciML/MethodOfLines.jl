function SciMLBase.discretize(pdesys::PDESystem,
                              discretization::MOLFiniteDifference{G},
                              analytic = nothing, kwargs...) where {G<:StaggeredGrid}
    @info "in staggered_discretize.jl:discretize()"
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    try
        simpsys = structural_simplify(sys)
        if tspan === nothing
            add_metadata!(get_metadata(sys), sys)
            return prob = NonlinearProblem(simpsys, ones(length(simpsys.states));
                                           discretization.kwargs..., kwargs...)
        else
            # Use ODAE if nessesary
            add_metadata!(get_metadata(simpsys), sys)
            prob = ODEProblem(simpsys, Pair[], tspan; discretization.kwargs...,
                              kwargs...)
            if analytic === nothing
                return prob
            else
                f = ODEFunction(pdesys, discretization, analytic = analytic,
                                discretization.kwargs..., kwargs...)

                prob = ODEProblem(f, prob.u0, prob.tspan, prob.p;
                                  discretization.kwargs..., kwargs...);

                len = floor(Int, length(prob.u0)/2);
                #@variables rho[1:len] phi[1:len]
                @variables rho[1:len] phi[1:len+1]
                drho = (prob.f([collect(rho); collect(phi)], nothing, 0.0)[1:len]);
                dphi = (prob.f([collect(rho); collect(phi)], nothing, 0.0)[len+1:end]);

                gen_drho = eval(Symbolics.build_function(drho, collect(rho), collect(phi))[2]);
                gen_dphi = eval(Symbolics.build_function(dphi, collect(rho), collect(phi))[2]);

                dynamical_f1(_drho,u,p,t) = gen_drho(_drho, u[1:len], u[len+1:end]);
                dynamical_f2(_dphi,u,p,t) = gen_dphi(_dphi, u[1:len], u[len+1:end]);
                u0 = [prob.u0[1:len]; prob.u0[len+1:end]];
                return DynamicalODEProblem(dynamical_f1, dynamical_f2, u0[1:len], u0[len+1:end], (0.0,1.0))
            end
        end
    catch e
        error_analysis(sys, e)
    end
end
