"""
# `cartesian_nonlinear_laplacian`

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
        expr, II, derivweights, s::DiscreteSpace, indexmap, bcmap, depvars, x, u
    )
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    # See scheme 1, namely the term without the 1/r dependence. See also #354 and #371 in DiffEqOperators, the previous home of this package.
    bs = filter_interfaces(bcmap[operation(u)][x])
    N = ndims(u, s)
    N == 0 && return 0
    jx = j, x = (x2i(s, u, x), x)

    D_inner = derivweights.halfoffsetmap[1][Differential(x)]
    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    inner_interpolater = derivweights.interpmap[x]

    # Get the outer weights and stencil. clip() essentially removes a point from either end of the grid, for this reason this function is only defined on the interior, not in bcs
    #* Need to see how to handle this with interface boundaries

    cliplen = length(s, x) - 1
    outerweights,
        outerstencil = get_half_offset_weights_and_stencil(
        D_outer, II - unitindex(N, j), s, bs, u, jx, cliplen
    )

    interface_wrap(stencil) = bwrap.(stencil, (bs,), (s,), (jx,))

    # Get the correct weights and stencils for this II
    interp_weights_and_stencil = [
        get_half_offset_weights_and_stencil(
                inner_interpolater, I, s, bs, u, jx
            )
            for I in outerstencil
    ]
    function deriv_weights_and_stencil(u, i, order)
        return get_half_offset_weights_and_stencil(
            derivweights.halfoffsetmap[1][Differential(x)^order],
            outerstencil[i], s, bs, u, (x2i(s, u, x), x)
        )
    end

    # map variables to symbolically inerpolated/extrapolated expressions
    function map_vars_to_interpolated(stencil, weights)
        return [v => sym_dot(weights, s.discvars[v][interface_wrap(stencil)]) for v in depvars]
    end

    # Map parameters to interpolated values. Using simplistic extrapolation/interpolation for now as grids are uniform
    #TODO: make this more efficient
    function map_ivs_to_interpolated(stencil, weights)
        return safe_vcat(
            [
                x => dot(
                    weights, getindex.((s.grid[x],), getindex.(interface_wrap(stencil), (j,)))
                ),
            ],
            [s.x̄[k] => s.grid[s.x̄[k]][II[k]] for k in setdiff(1:N, [j])]
        )
    end

    # Go ham and try to discretize anything that appears inside the nonlinear laplacian
    drules = generate_deriv_rules(
        II, s, depvars, derivweights, indexmap, bcmap, jx,
        outerstencil, deriv_weights_and_stencil, interface_wrap
    )

    # Symbolically interpolate the expression
    interpolated_expr = map(enumerate(interp_weights_and_stencil)) do (i, weights_stencil)
        weights, stencil = weights_stencil
        rules = vcat(
            mapreduce(d -> d(i), vcat, drules), map_vars_to_interpolated(stencil, weights),
            map_ivs_to_interpolated(stencil, weights)
        )
        substitute(expr * Differential(x)(u), rules)
    end

    # multiply the inner finite difference by the interpolated expression, and finally take the outer finite difference
    return sym_dot(outerweights, interpolated_expr)
end

function generate_deriv_rules(
        II, s::DiscreteSpace, depvars, derivweights, indexmap, bmap,
        jx, outerstencil, deriv_weights_and_stencil, interface_wrap
    )
    # Generate rules for the derivatives of the variables
    j, x = jx
    return mapreduce(vcat, depvars) do u
        mapreduce(vcat, ivs(u, s)) do y
            let orders = derivweights.orders[x]
                map(vcat(1, orders)) do order
                    # If we're differentiating with respect to the same variable, we need to use the correct weights and stencil
                    # for the order of the derivative
                    if isequal(x, y)
                        (i) -> begin
                            let (weights, stencil) = deriv_weights_and_stencil(u, i, order)
                                [
                                    (Differential(x)^order)(u) => sym_dot(
                                        weights, s.discvars[u][interface_wrap(stencil)]
                                    ),
                                ]
                            end
                        end
                    else
                        # Otherwise, we will use the usual rules shifted
                        (i) -> begin
                            let II = outerstencil[i]
                                central_deriv_rules_cartesian = generate_cartesian_rules(
                                    II, s, depvars, derivweights, bmap, indexmap, nothing
                                )
                                central_deriv_rules_cartesian
                            end
                        end
                    end
                end
            end
        end
    end
end

function replacevals(ex, s, u, depvars, II, indexmap)
    rules = valmaps(s, u, depvars, II, indexmap)
    return substitute(ex, rules)
end

@inline function generate_nonlinlap_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms
    )
    rules = reduce(
        safe_vcat,
        [
            vec(
                    [
                        @rule *(
                            ~~c,
                            $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)),
                            ~~d
                        ) => *(
                            ~c...,
                            cartesian_nonlinear_laplacian(
                                *(~a..., ~b...), Idx(II, s, u, indexmap),
                                derivweights, s, indexmap, bcmap, depvars, x, u
                            ),
                            ~d...
                        ) for x in ivs(u, s)
                    ]
                ) for u in depvars
        ],
        init = []
    )

    rules = safe_vcat(
        rules,
        reduce(
            safe_vcat,
            [
                vec(
                        [
                            @rule $(Differential(x))(
                                *(
                                    ~~a,
                                    $(Differential(x))(u),
                                    ~~b
                                )
                            ) => cartesian_nonlinear_laplacian(
                                *(~a..., ~b...), Idx(II, s, u, indexmap),
                                derivweights, s, indexmap, bcmap, depvars, x, u
                            ) for x in ivs(u, s)
                        ]
                    )
                    for u in depvars
            ],
            init = []
        )
    )

    rules = safe_vcat(
        rules,
        reduce(
            safe_vcat,
            [
                vec(
                        [
                            @rule (
                                $(Differential(x))(
                                    $(Differential(x))(u) /
                                    ~a
                                )
                            ) => cartesian_nonlinear_laplacian(
                                1 / ~a, Idx(II, s, u, indexmap), derivweights,
                                s, indexmap, bcmap, depvars, x, u
                            ) for x in ivs(u, s)
                        ]
                    )
                    for u in depvars
            ],
            init = []
        )
    )

    rules = safe_vcat(
        rules,
        reduce(
            safe_vcat,
            [
                vec(
                        [
                            @rule *(
                                ~~b,
                                ($(Differential(x))($(Differential(x))(u) / ~a)),
                                ~~c
                            ) => *(
                                ~b...,
                                ~c...,
                                cartesian_nonlinear_laplacian(
                                    1 / ~a, Idx(II, s, u, indexmap), derivweights,
                                    s, indexmap, bcmap, depvars, x, u
                                )
                            ) for x in ivs(u, s)
                        ]
                    )
                    for u in depvars
            ],
            init = []
        )
    )

    rules = safe_vcat(
        rules,
        reduce(
            safe_vcat,
            [
                vec(
                        [
                            @rule /(
                                *(~~b, ($(Differential(x))(*(~~a, $(Differential(x))(u), ~~d))), ~~c),
                                ~e
                            ) => /(
                                *(
                                    ~b...,
                                    ~c...,
                                    cartesian_nonlinear_laplacian(
                                        *(~a..., ~d...), Idx(II, s, u, indexmap),
                                        derivweights, s, indexmap, bcmap, depvars, x, u
                                    )
                                ),
                                replacevals(~e, s, u, depvars, II, indexmap)
                            ) for x in ivs(u, s)
                        ]
                    )
                    for u in depvars
            ],
            init = []
        )
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
