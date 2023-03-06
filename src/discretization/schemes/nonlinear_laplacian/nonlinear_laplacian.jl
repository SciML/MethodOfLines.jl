
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
function cartesian_nonlinear_laplacian(expr, II, derivweights, s::DiscreteSpace, bs, depvars, x, u)
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    # See scheme 1, namely the term without the 1/r dependence. See also #354 and #371 in DiffEqOperators, the previous home of this package.
    N = ndims(u, s)
    N == 0 && return 0
    jx = j, x = (x2i(s, u, x), x)

    D_inner = derivweights.halfoffsetmap[1][Differential(x)]
    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    inner_interpolater = derivweights.interpmap[x]

    # Get the outer weights and stencil. clip() essentially removes a point from either end of the grid, for this reason this function is only defined on the interior, not in bcs
    #* Need to see how to handle this with interface boundaries

    cliplen = length(s, x) - 1
    outerweights, outerstencil = get_half_offset_weights_and_stencil(D_outer, II - unitindex(N, j), s, bs, u, jx, cliplen)


    interface_wrap(stencil) = bwrap.(stencil, (bs,), (s,), (jx,))

    # Get the correct weights and stencils for this II
    inner_deriv_weights_and_stencil = [get_half_offset_weights_and_stencil(D_inner, I, s, bs, u, jx) for I in outerstencil]
    interp_weights_and_stencil = [get_half_offset_weights_and_stencil(inner_interpolater, I, s, bs, u, jx) for I in outerstencil]

    # map variables to symbolically inerpolated/extrapolated expressions
    map_vars_to_interpolated(stencil, weights) = [v => sym_dot(weights, s.discvars[v][interface_wrap(stencil)]) for v in depvars]

    # Map parameters to interpolated values. Using simplistic extrapolation/interpolation for now as grids are uniform
    #TODO: make this more efficient
    map_ivs_to_interpolated(stencil, weights) = safe_vcat([x => dot(weights, getindex.((s.grid[x],), getindex.(interface_wrap(stencil), (j,))))], [s.x̄[k] => s.grid[s.x̄[k]][II[k]] for k in setdiff(1:N, [j])])

    # Take the inner finite difference
    inner_difference = [sym_dot(inner_weights, s.discvars[u][interface_wrap(inner_stencil)]) for (inner_weights, inner_stencil) in inner_deriv_weights_and_stencil]

    # Symbolically interpolate the multiplying expression


    interpolated_expr = map(interp_weights_and_stencil) do (weights, stencil)
        substitute(substitute(expr, map_vars_to_interpolated(stencil, weights)), map_ivs_to_interpolated(stencil, weights))
    end

    # multiply the inner finite difference by the interpolated expression, and finally take the outer finite difference
    return sym_dot(outerweights, inner_difference .* interpolated_expr)
end

@inline function generate_nonlinlap_rules(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    rules = reduce(safe_vcat, [vec([@rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => *(~c..., cartesian_nonlinear_laplacian(*(~a..., ~b...), Idx(II, s, u, indexmap), derivweights, s, filter_interfaces(bcmap[operation(u)][x]), depvars, x, u), ~d...) for x in ivs(u, s)]) for u in depvars], init = [])

    rules = safe_vcat(rules, reduce(safe_vcat, [vec([@rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => cartesian_nonlinear_laplacian(*(~a..., ~b...), Idx(II, s, u, indexmap), derivweights, s, filter_interfaces(bcmap[operation(u)][x]), depvars, x, u) for x in ivs(u, s)]) for u in depvars], init = []))

    rules = safe_vcat(rules, reduce(safe_vcat, [vec([@rule ($(Differential(x))($(Differential(x))(u) / ~a)) => cartesian_nonlinear_laplacian(1 / ~a, Idx(II, s, u, indexmap), derivweights, s, filter_interfaces(bcmap[operation(u)][x]), depvars, x, u) for x in ivs(u, s)]) for u in depvars], init = []))

    rules = safe_vcat(rules, reduce(safe_vcat, [vec([@rule *(~~b, ($(Differential(x))($(Differential(x))(u) / ~a)), ~~c) => *(~b..., ~c..., cartesian_nonlinear_laplacian(1 / ~a, Idx(II, s, u, indexmap), derivweights, s, filter_interfaces(bcmap[operation(u)][x]), depvars, x, u)) for x in ivs(u, s)]) for u in depvars], init = []))

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
