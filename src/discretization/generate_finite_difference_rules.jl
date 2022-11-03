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

There is a genral catch all ruleset that uses the cartesian centered difference scheme for derivatives, and simply the discretized variable at the given gridpoint for particular variables.

There are of course more specific schemes that are used to improve stability/speed/accuracy when particular forms are encountered in the PDE. These rules are applied first to override the general ruleset.

##Currently implemented special cases are as follows:
    - Spherical derivatives
    - Nonlinear laplacian uses a half offset centered scheme for the inner derivative to improve stability
    - Spherical nonlinear laplacian.
    - Upwind schemes to be used for odd ordered derivatives multiplied by a coefficient, downwinding when the coefficient is positive, and upwinding when the coefficient is negative.

Please submit an issue if you know of any special cases which impact stability or accuracy that are not implemented, with links to papers and/or code that demonstrates the special case.
"""
function generate_finite_difference_rules(II::CartesianIndex, s::DiscreteSpace, depvars, pde::Equation, derivweights::DifferentialDiscretizer, pmap, indexmap)

    terms = split_terms(pde, s.x̄)

    # Standard cartesian centered difference scheme
    central_deriv_rules_cartesian = generate_cartesian_rules(II, s, depvars, derivweights, pmap, indexmap, terms)

    # Advection rules
    if derivweights.advection_scheme isa UpwindScheme
        advection_rules = generate_winding_rules(II, s, depvars, derivweights, pmap, indexmap, terms)
    elseif derivweights.advection_scheme isa WENOScheme
        advection_rules = generate_WENO_rules(II, s, depvars, derivweights, pmap, indexmap, terms)
    else
        error("Unsupported advection scheme $(derivweights.advection_scheme) encountered.")
    end

    # Nonlinear laplacian scheme
    nonlinlap_rules = generate_nonlinlap_rules(II, s, depvars, derivweights, pmap, indexmap, terms)

    # Spherical diffusion scheme
    spherical_diffusion_rules = generate_spherical_diffusion_rules(II, s, depvars, derivweights, pmap, indexmap, split_additive_terms(pde))

    return vcat(vec(spherical_diffusion_rules), vec(nonlinlap_rules), vec(central_deriv_rules_cartesian), vec(advection_rules))
end

#######################################################################################
# Stencil interface
#######################################################################################

function generate_finite_difference_rules(interior, s::DiscreteSpace, depvars,
                                          pde::Equation,
                                          derivweights::DifferentialDiscretizer, pmap,
                                          indexmap)

    terms = split_terms(pde, s.x̄)

    # Standard cartesian centered difference scheme
    central_deriv_rules_cartesian = generate_cartesian_rules(interior, s, depvars,
                                                             derivweights, pmap, indexmap,
                                                             terms)

    # Advection rules
    if derivweights.advection_scheme isa UpwindScheme
        advection_rules = generate_winding_rules(interior, s, depvars, derivweights, pmap,
                                                 indexmap, terms)
    elseif derivweights.advection_scheme isa WENOScheme
        advection_rules = generate_WENO_rules(interior, s, depvars, derivweights, pmap,
                                              indexmap, terms)
    else
        error("Unsupported advection scheme $(derivweights.advection_scheme) encountered.")
    end

    # Nonlinear laplacian scheme
    nonlinlap_rules = generate_nonlinlap_rules(interior, s, depvars, derivweights, pmap,
                                               indexmap, terms)

    # Spherical diffusion scheme
    spherical_diffusion_rules = generate_spherical_diffusion_rules(interior, s, depvars,
                                                                   derivweights, pmap,
                                                                   indexmap,
                                                                   split_additive_terms(pde))

    return vcat(vec(spherical_diffusion_rules), vec(nonlinlap_rules), vec(central_deriv_rules_cartesian), vec(advection_rules))
end
