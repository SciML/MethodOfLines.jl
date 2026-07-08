using ModelingToolkit: get_ps, mtkcompile, ProblemTypeCtx, getmetadata

function _safe_unwrap(x)
    return x isa Num ? unwrap(x) : x
end

function _build_ode_problem(
        simpsys, mol_metadata, tspan, discretization::MOLFiniteDifference, kwargs...
    )
    u0 = hasproperty(mol_metadata, :u0) ? mol_metadata.u0 : []
    pdesys_ic = mol_metadata.pdesys.initial_conditions
    ps_raw = get_ps(mol_metadata.pdesys)
    prob_kwargs = (;
        build_initializeprob = false,
        discretization.kwargs...,
        kwargs...,
    )
    if ps_raw !== nothing && ps_raw !== SciMLBase.NullParameters() && !isempty(ps_raw)
        param_vals = Dict{Any, Any}()
        if first(ps_raw) isa Pair
            for p in ps_raw
                param_vals[first(p)] = last(p)
            end
        else
            ps_unwrapped = [_safe_unwrap(p) for p in ps_raw]
            for (k, v) in pairs(pdesys_ic)
                k_unwrapped = _safe_unwrap(k)
                if any(p -> isequal(k_unwrapped, _safe_unwrap(p)), ps_unwrapped)
                    v_numeric = try
                        Symbolics.value(v)
                    catch
                        _safe_unwrap(v)
                    end
                    param_vals[k] = v_numeric
                end
            end
        end
        if !isempty(param_vals)
            op = merge(Dict(u0), param_vals)
            return ODEProblem(simpsys, op, tspan; prob_kwargs...)
        end
    end
    return ODEProblem(simpsys, u0, tspan; prob_kwargs...)
end

function SciMLBase.discretize(
        pdesys::PDESystem,
        discretization::MOLFiniteDifference{G, D};
        analytic = nothing, checks = true, kwargs...
    ) where {G, D <: ScalarizedDiscretization}
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization; checks = checks)
    return try
        simpsys = mtkcompile(sys)
        if tspan === nothing
            add_metadata!(getmetadata(sys, ProblemTypeCtx, nothing), sys)
            unknowns_list = ModelingToolkit.unknowns(simpsys)
            u0_guess = Dict(u => 1.0 for u in unknowns_list)
            return NonlinearProblem(
                simpsys, u0_guess;
                discretization.kwargs..., kwargs...
            )
        else
            mol_metadata = getmetadata(simpsys, ProblemTypeCtx, nothing)
            add_metadata!(mol_metadata, sys)
            prob = _build_ode_problem(simpsys, mol_metadata, tspan, discretization, kwargs...)
            if analytic === nothing
                return apply_dae_initialization_fallback(prob, discretization; kwargs...)
            else
                f = SciMLBase.ODEFunction(
                    pdesys, discretization, analytic = analytic,
                    discretization.kwargs..., kwargs...
                )
                prob = ODEProblem(
                    f, prob.u0, prob.tspan, prob.p;
                    discretization.kwargs..., kwargs...
                )
                return apply_dae_initialization_fallback(prob, discretization; kwargs...)
            end
        end
    catch e
        error_analysis(sys, e)
    end
end
