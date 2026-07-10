function _disc_kwargs_nt(disc::MOLFiniteDifference)
    kw = disc.kwargs
    return kw isa NamedTuple ? kw : NamedTuple()
end

function _user_provided_initializealg(
        disc::MOLFiniteDifference, discretize_kwargs::NamedTuple, prob
    )
    merged = merge(_disc_kwargs_nt(disc), discretize_kwargs)
    return haskey(merged, :initializealg) || haskey(prob.kwargs, :initializealg)
end

"""
    apply_dae_initialization_fallback(prob, discretization; kwargs...)

Remake with `BrownFullBasicInit()` when `is_implicit_dae(prob)` and no user `initializealg`.
"""
function apply_dae_initialization_fallback(
        prob::SciMLBase.ODEProblem,
        discretization::MOLFiniteDifference;
        kwargs...
    )
    discretize_kwargs = NamedTuple(kwargs)
    if _user_provided_initializealg(discretization, discretize_kwargs, prob)
        return prob
    end
    if is_implicit_dae(prob)
        return SciMLBase.remake(prob; initializealg = BrownFullBasicInit())
    end
    return prob
end

apply_dae_initialization_fallback(prob, ::MOLFiniteDifference; kwargs...) = prob
