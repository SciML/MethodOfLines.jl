# abstract type AbstractRule{T} end

# abstract type ConditionalRule{T} <: AbstractRule{T} end
# struct SimpleRule{T} <: AbstractReplacementRule{T}
#     r::T
#     priority::Int
# end

# struct UpwindRule{T, T2}
#     r::T
#     priority::Int
#     condition::T2
# end
# struct RuleSet{T, T2}
#     derivrules::T
#     valrules::T2
# end

# function RuleSet(rules::Vector{T}, conditional_rules::Vector{C}) where {T<:SimpleRule, C<:ConditionalRule}
#     priorities = vcat(map(r -> r.priority, rules), map(r -> r.priority, conditional_rules))

#     for (i,r) in enumerate(vcat(rules, conditional_rules))

#     end

# ModelingToolkit.substitute(expr, rule::AbstractRule{T}) where T = substitute(expr, rule.r)

"""
`generate_finite_difference_rules`

Generate a vector of finite difference rules to dictate what to replace variables in the `pde` with at the gridpoint `II`.

Care is taken to make sure that the rules only use points that are actually in the discretized grid by progressively up/downwinding the stencils when the gridpoint `II` is close to the boundary.

There is a general catch all ruleset that uses the cartesian centered difference scheme for derivatives, and simply the discretized variable at the given gridpoint for particular variables.

There are of course more specific schemes that are used to improve stability/speed/accuracy when particular forms are encountered in the PDE. These rules are applied first to override the general ruleset.

##Currently implemented special cases are as follows:
    - Spherical derivatives
    - Nonlinear laplacian uses a half offset centered scheme for the inner derivative to improve stability
    - Spherical nonlinear laplacian.
    - Upwind schemes to be used for odd ordered derivatives multiplied by a coefficient, downwinding when the coefficient is positive, and upwinding when the coefficient is negative.

Please submit an issue if you know of any special cases which impact stability or accuracy that are not implemented, with links to papers and/or code that demonstrates the special case.
"""
function generate_finite_difference_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, pde::Equation,
        derivweights::DifferentialDiscretizer, bmap, indexmap
    )
    terms = split_terms(pde, s.x̄)
    if length(II) != 0
        # Standard cartesian centered difference scheme
        central_deriv_rules_cartesian = generate_cartesian_rules(
            II, s, depvars, derivweights, bmap, indexmap, terms
        )
        # Mixed derivative rules
        mixed_deriv_rules_cartesian = generate_mixed_rules(
            II, s, depvars, derivweights, bmap, indexmap, terms
        )
        # Advection rules
        if derivweights.advection_scheme isa UpwindScheme
            advection_rules = generate_winding_rules(
                II, s, depvars, derivweights, bmap, indexmap, terms
            )
        elseif derivweights.advection_scheme isa FunctionalScheme
            advection_rules = generate_advection_rules(
                derivweights.advection_scheme, II, s,
                depvars, derivweights, bmap, indexmap, terms
            )
            winding_rules = generate_winding_rules(
                II, s, depvars, derivweights, bmap,
                indexmap, terms; skip = [1]
            )
            # Use append! instead of vcat to avoid allocation
            append!(advection_rules, winding_rules)
        else
            error("Unsupported advection scheme $(derivweights.advection_scheme) encountered.")
        end

        # Nonlinear laplacian scheme
        nonlinlap_rules = generate_nonlinlap_rules(
            II, s, depvars, derivweights, bmap, indexmap, terms
        )

        # Spherical diffusion scheme
        spherical_diffusion_rules = generate_spherical_diffusion_rules(
            II, s, depvars, derivweights, bmap, indexmap, split_additive_terms(pde)
        )
        integration_rules = generate_euler_integration_rules(
            II, s, depvars, indexmap, terms
        )
        # Flatten if needed
        integration_rules isa AbstractMatrix && (integration_rules = vec(integration_rules))
    else
        central_deriv_rules_cartesian = Pair[]
        advection_rules = Pair[]
        nonlinlap_rules = Pair[]
        spherical_diffusion_rules = Pair[]
        mixed_deriv_rules_cartesian = Pair[]
        integration_rules = Pair[]
    end

    cb_rules = generate_cb_rules(II, s, depvars, derivweights, bmap, indexmap, terms)

    whole_domain_rules = generate_whole_domain_integration_rules(II, s, depvars, indexmap, terms)
    whole_domain_rules isa AbstractMatrix && (whole_domain_rules = vec(whole_domain_rules))

    # Pre-allocate result array with estimated size to avoid repeated resizing
    result = Pair[]
    sizehint!(result,
        length(cb_rules) + length(spherical_diffusion_rules) + length(nonlinlap_rules) +
        length(mixed_deriv_rules_cartesian) + length(central_deriv_rules_cartesian) +
        length(advection_rules) + length(integration_rules) + length(whole_domain_rules)
    )

    # Use append! instead of vcat to avoid allocations
    _append_vec!(result, cb_rules)
    _append_vec!(result, spherical_diffusion_rules)
    _append_vec!(result, nonlinlap_rules)
    _append_vec!(result, mixed_deriv_rules_cartesian)
    _append_vec!(result, central_deriv_rules_cartesian)
    _append_vec!(result, advection_rules)
    _append_vec!(result, integration_rules)
    _append_vec!(result, whole_domain_rules)

    return result
end

# Helper to append vectors or matrices efficiently
@inline function _append_vec!(result, x)
    if x isa AbstractMatrix
        append!(result, vec(x))
    else
        append!(result, x)
    end
    return result
end

function generate_finite_difference_rules(
        II::CartesianIndex, s::DiscreteSpace{W, M, G}, depvars,
        pde::Equation, derivweights::DifferentialDiscretizer,
        bmap, indexmap
    ) where {W, M, G <: StaggeredGrid}
    terms = split_terms(pde, s.x̄)
    if length(II) != 0
        # Standard cartesian centered difference scheme
        central_deriv_rules_cartesian = generate_cartesian_rules(
            II, s, depvars, derivweights, bmap, indexmap, terms
        )
    else
        central_deriv_rules_cartesian = Pair[]
    end

    whole_domain_rules = generate_whole_domain_integration_rules(II, s, depvars, indexmap, terms)
    whole_domain_rules isa AbstractMatrix && (whole_domain_rules = vec(whole_domain_rules))

    # Pre-allocate and use append! instead of vcat
    result = Pair[]
    sizehint!(result, length(central_deriv_rules_cartesian) + length(whole_domain_rules))
    _append_vec!(result, central_deriv_rules_cartesian)
    _append_vec!(result, whole_domain_rules)

    return result
end
