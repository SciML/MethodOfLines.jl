function SciMLBase.discretize(
        pdesys::PDESystem,
        discretization::MOLFiniteDifference{G, D};
        analytic = nothing, checks = true, kwargs...
    ) where {G <: Union{CenterAlignedGrid, EdgeAlignedGrid}, D <: ScalarizedDiscretization}
    prob = invoke(
        SciMLBase.discretize,
        Tuple{PDESystem, AbstractEquationSystemDiscretization},
        pdesys, discretization;
        analytic, checks, kwargs...
    )
    return apply_dae_initialization_fallback(prob, discretization; kwargs...)
end
