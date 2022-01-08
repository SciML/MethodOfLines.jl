function calculate_weights_spherical(order::Int, x0::T, x::AbstractVector, idxs::AbstractVector) where T<:Real
    # TODO: use Fornberg
    # Spherical domain: see #367
    # https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    # Only order 2 is implemented
    @assert order == 2
    # Only 2nd order discretization is implemented
    # We can't activate this assertion for now because the rules try to create the spherical Laplacian
    # before checking whether there is a spherical Laplacian
    # this could be fixed by dispatching on domain type when we have different domain types
    # but for now everything is an Interval
    # @assert length(x) == 3
    i = idxs[2]
    dx1 = x[i] - x[i-1]
    dx2 = x[i+1] - x[i]
    i0 = i - 1 # indexing starts at 0 in the paper and starts at 1 in julia
    1 / (i0 * dx1 * dx2) * [i0-1, -2i0, i0+1]
end

"""
`interpolate_discrete_param`

Interpolate gridpoints by taking the average of the values of the discrete points, or if the offset is outside the grid, extrapolate the value with dx.
"""
function interpolate_discrete_param(II, s, itap, j, x)
    # * This will need to be updated to dispatch on grid type when grids become more general
    offset = itap+1/2
    if (II[j]+itap) < 1
        return s.grid[x][1]+s.dxs[x]*offset
    elseif (II[j]+itap) > (length(x) -  1)
        return s.grid[x][length(x)]+s.dxs[x]*offset
    else
        return (s.grid[x][II[j]+offset]+s.grid[x][II[j]+offset+1])/2
    end
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
function cartesian_nonlinear_laplacian(expr, II, derivweights, s, x, u)
    jx = j, x = (s.x2i(x), x)
    D_outer = derivweights.map[Differential(x)]
    inner_interpolater, D_inner = derivweights.nonlinlapmap[x]

    weights = D_outer.stencil_coefs
    # Index offsets of each stencil in the inner finite difference to get the correct stencil for each needed half grid point, 0 corresopnds to x+1/2
    itaps = half_range(D_outer.stencil_length) .- 1
    
    # Get the correct weights and stencils for this II
    inner_deriv_weights_and_stencil = [get_weights_and_stencil(D_inner, II, s, itap, i, jx) for (i,itap) in enumerate(itaps)]
    interp_weights_and_stencil = [get_weights_and_stencil(inner_interpolater, II, s, itap, i, jx) for (i,itap) in enumerate(itaps)]

    # map variables to symbolically inerpolated/extrapolated expressions
    map_vars_to_interpolated(stencil, weights) = [v => dot(weights, s.discvars[v][stencil]) for v in s.vars]

    # Map parameters to interpolated values. Using simplistic extrapolation/interpolation for now as grids are uniform
    map_params_to_interpolated(itap) = [z => interpolate_discrete_param(II, s, itap, i, z) for (i,z) in s.nottime]

    # Take the inner finite difference
    inner_difference = [dot(inner_weights, s.discvars[u][inner_stencil]) for (inner_weights, inner_stencil) in inner_deriv_weights_and_stencil]
    
    # Symbolically interpolate the multiplying expression
    interpolated_expr = [Num(substitute(substitute(expr, map_vars_to_interpolated(stencil, weights)), map_params_to_interpolated(itap))) 
                        for (itap,(weights, stencil)) in zip(itaps, interp_weights_and_stencil)]
 
    # multiply the inner finite difference by the interpolated expression, and finally take the outer finite difference
    return dot(weights, inner_difference .* interpolated_expr)
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

##Planned special cases include:
    - Up/Downwind schemes to be used for odd ordered derivatives multiplied by a coefficient, downwinding when the coefficient is positive, and upwinding when the coefficient is negative.

Please submit an issue if you know of any special cases that are not implemented, with links to papers and/or code that demonstrates the special case.
"""
function generate_finite_difference_rules(II, s, pde, derivweights)

    valrules = vcat([u => s.discvars[u][II] for u in s.vars],
                    [x => s.grid[x][II[j]] for x in params(s)])
    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]
    central_deriv_rules_cartesian = Array{Pair{Num,Num},1}()
    for x in s.nottime
        j = s.x2i[x]
        rs = [(Differential(x)^d)(u) => central_deriv_cartesian(derivweights.map[Differential(x)^d], II, s, (j,x), u) for d in derivweights.orders[x], u in s.vars]

        central_deriv_rules_cartesian = vcat(central_deriv_rules_cartesian, rs)
    end
    central_deriv_rules_spherical = [Differential(x)(x^2*Differential(x)(u))/x^2 => central_deriv_spherical(II,j,k)
                                    for (j,x) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]


    # TODO: upwind rules needs interpolation into `@rule`
    #forward_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j],II[j]+1]])
    #reverse_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j]-1,II[j]]])
    # upwinding_rules = [@rule(*(~~a,$(Differential(s.nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
    #                         *(~~a..., ~~b..., dot(reverse_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[1:2]])),
    #                         *(~~a..., ~~b..., dot(forward_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[2:3]]))))
    #                         for j in 1:nparams(s), k in 1:length(pdesys.s.vars)]

    ## Discretization of non-linear laplacian.


    cartesian_deriv_rules = [@rule ($(Differential(x))(*(~~a, $(Differential(x))(u), ~~b))) => cartesian_nonlinear_laplacian(*(a..., b...), II, derivweights, s, x, u) for x in s.nottime, u in s.vars]

    cartesian_deriv_rules = vcat(vec(cartesian_deriv_rules),vec(
                            [@rule ($(Differential(x))($(Differential(x))(u)/~a)) => cartesian_nonlinear_laplacian(1/~a, II, derivweights, s, x, u) for x in s.nottime, u in s.vars]))

    spherical_deriv_rules = [@rule *(~~a, ($(Differential(iv))((iv^2)*$(Differential(iv))(dv))), ~~b) / (iv^2) =>
                                *(~a..., central_deriv_spherical(II, j, k), ~b...)
                                for (j, iv) in enumerate(s.nottime), (k, dv) in enumerate(s.vars)]

    # r^-2 needs to be handled separately
    spherical_deriv_rules = vcat(vec(spherical_deriv_rules),vec(
                            [@rule *(~~a, (iv^-2) * ($(Differential(iv))((iv^2)*$(Differential(iv))(dv))), ~~b) =>
                                *(~a..., central_deriv_spherical(II, j, k), ~b...)
                                for (j, iv) in enumerate(s.nottime), (k, dv) in enumerate(s.vars)]))

    rhs_arg = istree(pde.rhs) && (SymbolicUtils.operation(pde.rhs) == +) ? SymbolicUtils.arguments(pde.rhs) : [pde.rhs]
    lhs_arg = istree(pde.lhs) && (SymbolicUtils.operation(pde.lhs) == +) ? SymbolicUtils.arguments(pde.lhs) : [pde.lhs]
    nonlinlap_rules = []
    for t in vcat(lhs_arg,rhs_arg)
        for r in cartesian_deriv_rules
            if r(t) !== nothing
                push!(nonlinlap_rules, t => r(t))
            end
        end
        for r in spherical_deriv_rules
            if r(t) !== nothing
                push!(nonlinlap_rules, t => r(t))
            end
        end
    end

    rules = vcat(vec(nonlinlap_rules),
                vec(central_deriv_rules_cartesian),
                vec(central_deriv_rules_spherical),
                valrules)
    return rules
end