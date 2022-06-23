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
`interpolate_discrete_param`

Interpolate gridpoints by taking the average of the values of the discrete points, or if the offset is outside the grid, extrapolate the value with dx.
"""
@inline function interpolate_discrete_param(i, s, itap, x, bpc)

    return s.grid[x][i+itap] + s.dxs[x] * 0.5

end

"""
`cartesian_nonlinear_laplacian`

Differential(x)(expr(x)*Differential(x)(u(x)))

Given an internal multiplying expression `expr`, return the correct finite difference equation for the nonlinear laplacian at the location in the grid given by `II`.

The inner derivative is discretized with the half offset centered scheme, giving the derivative at interpolated grid points offset by dx/2 from the regular grid.

The outer derivative is discretized with the centered scheme, giving the nonlinear laplacian at the grid point `II`.
For first order returns something like this:
`d/dx( a du/dx ) ~ (a(x+1/2) * (u[i+1] - u[i]) - a(x-1/2) * (u[i] - u[i-1]) / dx^2`

For 4th order, returns something like this:
```
first_finite_diffs = [a(x-3/2)*finitediff(u, i-3/2),
                      a(x-1/2)*finitediff(u, i-1/2),
                      a(x+1/2)*finitediff(u, i+1/2),
                      a(x+3/2)*finitediff(u, i+3/2)]

dot(central_finite_diff_weights, first_finite_diffs)
```

where `finitediff(u, i)` is the finite difference at the interpolated point `i` in the grid.

And so on.
"""
function cartesian_nonlinear_laplacian(
    expr,
    II,
    derivweights,
    s::DiscreteSpace{N},
    b,
    depvars,
    x,
    u,
) where {N}
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    # See scheme 1, namely the term without the 1/r dependence. See also #354 and #371 in DiffEqOperators, the previous home of this package.
    ndims(u, s) == 0 && return Num(0)
    jx = j, x = (x2i(s, u, x), x)
    @assert II[j] != 1 "The nonlinear laplacian is only defined on the interior of the grid, it is unsupported in boundary conditions."
    @assert II[j] != length(s, x) "The nonlinear laplacian is only defined on the interior of the grid, it is unsupported in boundary conditions."

    D_inner = derivweights.halfoffsetmap[1][Differential(x)]
    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    inner_interpolater = derivweights.interpmap[x]

    # Get the outer weights and stencil. clip() essentially removes a point from either end of the grid, for this reason this function is only defined on the interior, not in bcs#
    cliplen = length(s, x) - 1

    outerweights, outerstencil = get_half_offset_weights_and_stencil(
        D_outer,
        II - unitindex(N, j),
        s,
        b,
        u,
        jx,
        cliplen,
    )

    # Get the correct weights and stencils for this II
    inner_deriv_weights_and_stencil =
        [get_half_offset_weights_and_stencil(D_inner, I, s, b, u, jx) for I in outerstencil]
    interp_weights_and_stencil = [
        get_half_offset_weights_and_stencil(inner_interpolater, I, s, b, u, jx) for
        I in outerstencil
    ]

    # map variables to symbolically inerpolated/extrapolated expressions
    map_vars_to_interpolated(stencil, weights) =
        [v => dot(weights, s.discvars[v][stencil]) for v in depvars]

    # Map parameters to interpolated values. Using simplistic extrapolation/interpolation for now as grids are uniform
    #TODO: make this more efficient
    map_params_to_interpolated(stencil, weights) = vcat(
        [x => dot(weights, getindex.((s.grid[x],), getindex.(stencil, (j,))))],
        [s.x̄[k] => s.grid[s.x̄[k]][II[k]] for k in setdiff(1:N, [j])],
    )

    # Take the inner finite difference
    inner_difference = [
        dot(inner_weights, s.discvars[u][inner_stencil]) for
        (inner_weights, inner_stencil) in inner_deriv_weights_and_stencil
    ]

    # Symbolically interpolate the multiplying expression


    interpolated_expr = map(interp_weights_and_stencil) do (weights, stencil)
        Num(
            substitute(
                substitute(expr, map_vars_to_interpolated(stencil, weights)),
                map_params_to_interpolated(stencil, weights),
            ),
        )
    end

    # multiply the inner finite difference by the interpolated expression, and finally take the outer finite difference
    return dot(outerweights, inner_difference .* interpolated_expr)
end

"""
`spherical_diffusion`

Based on https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf

See scheme 1 in appendix A. The r = 0 case is treated in a later appendix
"""
function spherical_diffusion(innerexpr, II, derivweights, s, b, depvars, r, u)
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    D_1 = derivweights.map[Differential(r)]
    D_2 = derivweights.map[Differential(r)^2]

    #TODO!: Update this to use indvars of the pde
    # What to replace parameter x with given I
    _rsubs(x, I) = x => s.grid[x][I[s.x2i[x]]]
    # Full rules for substituting parameters in the inner expression
    rsubs(I) = vcat([v => s.discvars[v][I] for v in depvars], [_rsubs(x, I) for x in s.x̄])
    # Discretization func for u
    ufunc_u(v, I, x) = s.discvars[v][I]

    # 2nd order finite difference in u
    exprhere = Num(substitute(innerexpr, rsubs(II)))
    # Catch the r ≈ 0 case
    if Symbolics.unwrap(substitute(r, _rsubs(r, II))) ≈ 0
        D_2_u = central_difference(D_2, II, s, b, (s.x2i[r], r), u, ufunc_u)
        return 3exprhere * D_2_u # See appendix B of the paper
    end
    D_1_u = central_difference(D_1, II, s, b, (s.x2i[r], r), u, ufunc_u)
    # See scheme 1 in appendix A of the paper

    return exprhere * (
        D_1_u / Num(substitute(r, _rsubs(r, II))) +
        cartesian_nonlinear_laplacian(innerexpr, II, derivweights, s, b, depvars, r, u)
    )
end

@inline function generate_cartesian_rules(
    II::CartesianIndex,
    s::DiscreteSpace,
    depvars,
    derivweights::DifferentialDiscretizer,
    pmap,
    indexmap,
    terms,
)
    central_ufunc(u, I, x) = s.discvars[u][I]
    return reduce(
        vcat,
        [
            reduce(
                vcat,
                [
                    [
                        (Differential(x)^d)(u) => central_difference(
                            derivweights.map[Differential(x)^d],
                            Idx(II, s, u, indexmap),
                            s,
                            pmap.map[operation(u)][x],
                            (x2i(s, u, x), x),
                            u,
                            central_ufunc,
                        ) for d in (
                            let orders = derivweights.orders[x]
                                orders[iseven.(orders)]
                            end
                        )
                    ] for x in params(u, s)
                ],
            ) for u in depvars
        ],
    )
end

@inline function upwind_difference(
    expr,
    d::Int,
    II::CartesianIndex,
    s::DiscreteSpace,
    b,
    depvars,
    derivweights,
    (j, x),
    u,
    central_ufunc,
    indexmap,
)
    # TODO: Allow derivatives in expr
    expr = substitute(
        expr,
        valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap),
    )
    IfElse.ifelse(
        expr > 0,
        expr * upwind_difference(d, II, s, b, derivweights, (j, x), u, central_ufunc, true),
        expr *
        upwind_difference(d, II, s, b, derivweights, (j, x), u, central_ufunc, false),
    )
end

@inline function generate_winding_rules(
    II::CartesianIndex,
    s::DiscreteSpace,
    depvars,
    derivweights::DifferentialDiscretizer,
    pmap,
    indexmap,
    terms,
)
    wind_ufunc(v, I, x) = s.discvars[v][I]
    # for all independent variables and dependant variables
    rules = vcat(#Catch multiplication
        reduce(
            vcat,
            [
                reduce(
                    vcat,
                    [
                        [
                            @rule *(~~a, $(Differential(x)^d)(u), ~~b) =>
                                upwind_difference(
                                    *(~a..., ~b...),
                                    d,
                                    Idx(II, s, u, indexmap),
                                    s,
                                    pmap.map[operation(u)][x],
                                    depvars,
                                    derivweights,
                                    (x2i(s, u, x), x),
                                    u,
                                    wind_ufunc,
                                    indexmap,
                                ) for d in (
                                let orders = derivweights.orders[x]
                                    orders[isodd.(orders)]
                                end
                            )
                        ] for x in params(u, s)
                    ],
                ) for u in depvars
            ],
        ),

        #Catch division and multiplication, see issue #1
        reduce(
            vcat,
            [
                reduce(
                    vcat,
                    [
                        [
                            @rule /(*(~~a, $(Differential(x)^d)(u), ~~b), ~c) =>
                                upwind_difference(
                                    *(~a..., ~b...) / ~c,
                                    d,
                                    Idx(II, s, u, indexmap),
                                    s,
                                    pmap.map[operation(u)][x],
                                    depvars,
                                    derivweights,
                                    (x2i(s, u, x), x),
                                    u,
                                    wind_ufunc,
                                    indexmap,
                                ) for d in (
                                let orders = derivweights.orders[x]
                                    orders[isodd.(orders)]
                                end
                            )
                        ] for x in params(u, s)
                    ],
                ) for u in depvars
            ],
        ),
    )

    wind_rules = []

    # wind_exprs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(wind_rules, t => r(t))
            end
        end
    end

    return vcat(
        wind_rules,
        vec(
            mapreduce(vcat, depvars) do u
                mapreduce(vcat, params(u, s)) do x
                    j = x2i(s, u, x)
                    let orders = derivweights.orders[x]
                        oddorders = orders[isodd.(orders)]
                        # for all odd orders
                        if length(oddorders) > 0
                            map(oddorders) do d
                                (Differential(x)^d)(u) => upwind_difference(
                                    d,
                                    Idx(II, s, u, indexmap),
                                    s,
                                    pmap.map[operation(u)][x],
                                    derivweights,
                                    (j, x),
                                    u,
                                    wind_ufunc,
                                    true,
                                )
                            end
                        else
                            []
                        end
                    end
                end
            end,
        ),
    )


end

@inline function generate_nonlinlap_rules(
    II::CartesianIndex,
    s::DiscreteSpace,
    depvars,
    derivweights::DifferentialDiscretizer,
    pmap,
    indexmap,
    terms,
)
    rules = reduce(
        vcat,
        [
            vec([
                @rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => *(
                    ~c...,
                    cartesian_nonlinear_laplacian(
                        *(a..., b...),
                        Idx(II, s, u, indexmap),
                        derivweights,
                        s,
                        pmap.map[operation(u)][x],
                        depvars,
                        x,
                        u,
                    ),
                    ~d...,
                ) for x in params(u, s)
            ]) for u in depvars
        ],
    )

    rules = vcat(
        rules,
        reduce(
            vcat,
            [
                vec([
                    @rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) =>
                        cartesian_nonlinear_laplacian(
                            *(a..., b...),
                            Idx(II, s, u, indexmap),
                            derivweights,
                            s,
                            pmap.map[operation(u)][x],
                            depvars,
                            x,
                            u,
                        ) for x in params(u, s)
                ]) for u in depvars
            ],
        ),
    )

    rules = vcat(
        rules,
        reduce(
            vcat,
            [
                vec([
                    @rule ($(Differential(x))($(Differential(x))(u) / ~a)) =>
                        cartesian_nonlinear_laplacian(
                            1 / ~a,
                            Idx(II, s, u, indexmap),
                            derivweights,
                            s,
                            pmap.map[operation(u)][x],
                            depvars,
                            x,
                            u,
                        ) for x in params(u, s)
                ]) for u in depvars
            ],
        ),
    )

    rules = vcat(
        rules,
        reduce(
            vcat,
            [
                vec([
                    @rule *(~~b, ($(Differential(x))($(Differential(x))(u) / ~a)), ~~c) => *(
                        b...,
                        c...,
                        cartesian_nonlinear_laplacian(
                            1 / ~a,
                            Idx(II, s, u, indexmap),
                            derivweights,
                            s,
                            pmap.map[operation(u)][x],
                            depvars,
                            x,
                            u,
                        ),
                    ) for x in params(u, s)
                ]) for u in depvars
            ],
        ),
    )

    nonlinlap_rules = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(nonlinlap_rules, t => r(t))
            end
        end
    end
    return nonlinlap_rules
end

@inline function generate_spherical_diffusion_rules(
    II::CartesianIndex,
    s::DiscreteSpace,
    depvars,
    derivweights::DifferentialDiscretizer,
    pmap,
    indexmap,
    terms,
)
    rules = reduce(
        vcat,
        [
            vec([
                @rule *(
                    ~~a,
                    1 / (r^2),
                    ($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))),
                    ~~b,
                ) => *(
                    ~a...,
                    spherical_diffusion(
                        *(~c..., ~d..., ~e...),
                        Idx(II, s, u, indexmap),
                        derivweights,
                        s,
                        pmap.map[operation(u)][r],
                        depvars,
                        r,
                        u,
                    ),
                    ~b...,
                ) for r in params(u, s)
            ]) for u in depvars
        ],
    )

    rules = vcat(
        rules,
        reduce(
            vcat,
            [
                vec([
                    @rule /(
                        *(
                            ~~a,
                            $(Differential(r))(
                                *(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e),
                            ),
                            ~~b,
                        ),
                        (r^2),
                    ) => *(
                        ~a...,
                        ~b...,
                        spherical_diffusion(
                            *(~c..., ~d..., ~e...),
                            Idx(II, s, u, indexmap),
                            derivweights,
                            s,
                            pmap.map[operation(u)][r],
                            depvars,
                            r,
                            u,
                        ),
                    ) for r in params(u, s)
                ]) for u in depvars
            ],
        ),
    )

    rules = vcat(
        rules,
        reduce(
            vcat,
            [
                vec([
                    @rule /(
                        ($(Differential(r))(
                            *(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e),
                        )),
                        (r^2),
                    ) => spherical_diffusion(
                        *(~c..., ~d..., ~e...),
                        Idx(II, s, u, indexmap),
                        derivweights,
                        s,
                        pmap.map[operation(u)][r],
                        depvars,
                        r,
                        u,
                    ) for r in params(u, s)
                ]) for u in depvars
            ],
        ),
    )

    spherical_diffusion_rules = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(spherical_diffusion_rules, t => r(t))
            end
        end
    end
    return spherical_diffusion_rules
end

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
    - Up/Downwind schemes to be used for odd ordered derivatives multiplied by a coefficient, downwinding when the coefficient is positive, and upwinding when the coefficient is negative.

Please submit an issue if you know of any special cases which impact stability or accuracy that are not implemented, with links to papers and/or code that demonstrates the special case.
"""
function generate_finite_difference_rules(
    II::CartesianIndex,
    s::DiscreteSpace,
    depvars,
    pde::Equation,
    derivweights::DifferentialDiscretizer,
    pmap,
    indexmap,
)

    terms = split_terms(pde, s.x̄)

    # Standard cartesian centered difference scheme
    central_deriv_rules_cartesian =
        generate_cartesian_rules(II, s, depvars, derivweights, pmap, indexmap, terms)

    # Nonlinear laplacian scheme
    nonlinlap_rules =
        generate_nonlinlap_rules(II, s, depvars, derivweights, pmap, indexmap, terms)

    # Because winding needs to know about multiplying terms, we can't split the terms into additive and multiplicative terms.
    winding_rules =
        generate_winding_rules(II, s, depvars, derivweights, pmap, indexmap, terms)

    # Spherical diffusion scheme
    spherical_diffusion_rules = generate_spherical_diffusion_rules(
        II,
        s,
        depvars,
        derivweights,
        pmap,
        indexmap,
        split_additive_terms(pde),
    )

    return vcat(
        vec(spherical_diffusion_rules),
        vec(nonlinlap_rules),
        vec(winding_rules),
        vec(central_deriv_rules_cartesian),
    )
end
