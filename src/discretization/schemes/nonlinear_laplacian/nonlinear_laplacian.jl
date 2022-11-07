
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
function cartesian_nonlinear_laplacian(expr, II, derivweights, s::DiscreteSpace{N}, b, depvars, x, u) where {N}
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

    outerweights, outerstencil = get_half_offset_weights_and_stencil(D_outer, II - unitindex(N, j), s, b, u, jx, cliplen)

    # Get the correct weights and stencils for this II
    inner_deriv_weights_and_stencil = [get_half_offset_weights_and_stencil(D_inner, I, s, b, u, jx) for I in outerstencil]
    interp_weights_and_stencil = [get_half_offset_weights_and_stencil(inner_interpolater, I, s, b, u, jx) for I in outerstencil]

    # map variables to symbolically inerpolated/extrapolated expressions
    map_vars_to_interpolated(stencil, weights) = [v => dot(weights, s.discvars[v][stencil]) for v in depvars]

    # Map parameters to interpolated values. Using simplistic extrapolation/interpolation for now as grids are uniform
    #TODO: make this more efficient
    map_params_to_interpolated(stencil, weights) = vcat([x => dot(weights, getindex.((s.grid[x],), getindex.(stencil, (j,))))], [s.x̄[k] => s.grid[s.x̄[k]][II[k]] for k in setdiff(1:N, [j])])

    # Take the inner finite difference
    inner_difference = [dot(inner_weights, s.discvars[u][inner_stencil]) for (inner_weights, inner_stencil) in inner_deriv_weights_and_stencil]

    # Symbolically interpolate the multiplying expression


    interpolated_expr = map(interp_weights_and_stencil) do (weights, stencil)
        Num(substitute(substitute(expr, map_vars_to_interpolated(stencil, weights)), map_params_to_interpolated(stencil, weights)))
    end

    # multiply the inner finite difference by the interpolated expression, and finally take the outer finite difference
    return dot(outerweights, inner_difference .* interpolated_expr)
end

@inline function generate_nonlinlap_rules(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, pmap, indexmap, terms)
    rules = reduce(vcat, [vec([@rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => *(~c..., cartesian_nonlinear_laplacian(*(a..., b...), Idx(II, s, u, indexmap), derivweights, s, pmap.map[operation(u)][x], depvars, x, u), ~d...) for x in params(u, s)]) for u in depvars])

    rules = vcat(rules, reduce(vcat, [vec([@rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => cartesian_nonlinear_laplacian(*(a..., b...), Idx(II, s, u, indexmap), derivweights, s, pmap.map[operation(u)][x], depvars, x, u) for x in params(u, s)]) for u in depvars]))

    rules = vcat(rules, reduce(vcat, [vec([@rule ($(Differential(x))($(Differential(x))(u) / ~a)) => cartesian_nonlinear_laplacian(1 / ~a, Idx(II, s, u, indexmap), derivweights, s, pmap.map[operation(u)][x], depvars, x, u) for x in params(u, s)]) for u in depvars]))

    rules = vcat(rules, reduce(vcat, [vec([@rule *(~~b, ($(Differential(x))($(Differential(x))(u) / ~a)), ~~c) => *(b..., c..., cartesian_nonlinear_laplacian(1 / ~a, Idx(II, s, u, indexmap), derivweights, s, pmap.map[operation(u)][x], depvars, x, u)) for x in params(u, s)]) for u in depvars]))

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

########################################################################################
# Stencil interface
########################################################################################

#TODO: Decouple this from old implementation

function cartesian_nonlinear_laplacian(expr, interior, derivweights, s::DiscreteSpace{N}, b, depvars, x, u) where {N}
    args = params(u, s)
    ranges = map(x -> axes(s.grid[x])[1], args)
    interior = map(x -> interior[x], args)
    is = map(x -> s.index_syms[x], args)

    II = CartesianIndex(is...)

    deriv_expr = cartesian_nonlinear_laplacian(expr, II, derivweights, s, b, depvars, x, u)

    return FillArrayOp(deriv_expr, is, interior)
end

@inline function generate_nonlinlap_rules(interior, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, pmap, indexmap, terms)
    rules = reduce(vcat, [vec([@rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => *(~c..., cartesian_nonlinear_laplacian(*(a..., b...), interior, derivweights, s, pmap.map[operation(u)][x], depvars, x, u), ~d...) for x in params(u, s)]) for u in depvars])

    rules = vcat(rules, reduce(vcat, [vec([@rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => cartesian_nonlinear_laplacian(*(a..., b...), interior, derivweights, s, pmap.map[operation(u)][x], depvars, x, u) for x in params(u, s)]) for u in depvars]))

    rules = vcat(rules, reduce(vcat, [vec([@rule ($(Differential(x))($(Differential(x))(u) / ~a)) => cartesian_nonlinear_laplacian(1 / ~a, interior, derivweights, s, pmap.map[operation(u)][x], depvars, x, u) for x in params(u, s)]) for u in depvars]))

    rules = vcat(rules, reduce(vcat, [vec([@rule *(~~b, ($(Differential(x))($(Differential(x))(u) / ~a)), ~~c) => *(b..., c..., cartesian_nonlinear_laplacian(1 / ~a, interior, derivweights, s, pmap.map[operation(u)][x], depvars, x, u)) for x in params(u, s)]) for u in depvars]))

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
