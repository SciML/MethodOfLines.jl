function calculate_weights_cartesian(order::Int, x0::T, xs::AbstractVector, idxs::AbstractVector) where T<:Real
    # Cartesian domain: use Fornberg
    calculate_weights(order, x0, vec(xs[idxs]))
end
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
    # TODO: nonlinear diffusion in a spherical domain
    i = idxs[2]
    dx1 = x[i] - x[i-1]
    dx2 = x[i+1] - x[i]
    i0 = i - 1 # indexing starts at 0 in the paper and starts at 1 in julia
    1 / (i0 * dx1 * dx2) * [i0-1, -2i0, i0+1]
end

function central_deriv_cartesian(derivweights, II, s, jx, u, d)
    j, x = jx
    D = derivweights.map[Differential(x)^d]
    # unit index in direction of the derivative
    I1 = unitindices(nparams(s))[j] 
    # offset is offset due to boundary proximity
    if II[j] <= D.boundary_point_count
        weights = D.low_boundary_coefs[II[j]]    
        offset = D.boundary_point_count - II[j] + 1
    elseif II[j] > length(x) - D.boundary_point_count
        weights = D.high_boundary_coefs[length(s.grid)[j]-II[j]+1]
        offset = length(x) - II[j] - D.boundary_point_count
    else
        weights = D.stencil_coefs
        offset = 0
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.
    Itap = [II + (i+offset)*I1 for i in -D.boundary_point_count:D.boundary_point_count]

    return dot(weights, s.discvars[u][Itap])
end


# ! Update this to use dictionaries
# ! Create the stencil convolver function
# ! Work out how to walk the function tree and get the correct stencil
# ! While you're at it catch the nonlinear case
# ! Do interpolation with stencils
# * Find a more general stencil -> variable paradigm for the convolver, allowing an algebra of operators on their weights and tap points
function generate_finite_difference_rules(II, s, pde, discretization, derivweights)
    approx_order = discretization.centered_order
    I1 = oneunit(first(s.Igrid))
    Imin(order) = first(s.Igrid) + I1 * (order ÷ 2)
    Imax(order) = last(s.Igrid) - I1 * (order ÷ 2)
    stencil(j, order) = CartesianIndices(Tuple(map(x -> -x:x, (1:length(s.nottime) .== j) * (order ÷ 2))))
    # Use max and min to apply buffers
    central_neighbor_idxs(II,j,order) = stencil(j,order) .+ max(Imin(order),min(II,Imax(order)))

    # spherical Laplacian has a hardcoded order of 2 (only 2nd order is implemented)
    # both for derivative order and discretization order
    central_weights_spherical(II,x) = calculate_weights_spherical(2, s.grid[x][II[s.nottime2i[x]]], s.grid[x], vec(map(i->i[j], central_neighbor_idxs(II,s.x2i[x],2))))
    central_deriv_spherical(II,x,u) = dot(central_weights_spherical(II,x),s.discvars[u][central_neighbor_idxs(II,s.x2i[x],2)])

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]
    central_deriv_rules_cartesian = Array{Pair{Num,Num},1}()
    for (j,x) in enumerate(s.nottime)
        rs = [(Differential(x)^d)(u) => central_deriv_cartesian(derivweights, II, s, (j,x), u, d) for d in derivweights.orders, u in s.vars]
        for r in rs
            push!(central_deriv_rules_cartesian, r)
        end
    end
    # ! Catch this with stencil convolution
    central_deriv_rules_spherical = [Differential(x)(x^2*Differential(x)(u))/x^2 => central_deriv_spherical(II,j,k)
                                    for (j,x) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]

    valrules = vcat([u => s.discvars[u][II] for u in s.vars],
                    [x => s.grid[x][II[j]] for x in params(s)])

    # TODO: upwind rules needs interpolation into `@rule`
    forward_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j],II[j]+1]])
    reverse_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j]-1,II[j]]])
    # upwinding_rules = [@rule(*(~~a,$(Differential(s.nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
    #                         *(~~a..., ~~b..., dot(reverse_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[1:2]])),
    #                         *(~~a..., ~~b..., dot(forward_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[2:3]]))))
    #                         for j in 1:nparams(s), k in 1:length(pdesys.s.vars)]

    ## Discretization of non-linear laplacian.
    # d/dx( a du/dx ) ~ (a(x+1/2) * (u[i+1] - u[i]) - a(x-1/2) * (u[i] - u[i-1]) / dx^2
    reverse_finite_difference(II, j, u) = -dot(reverse_weights(II, j), s.discvars[u][central_neighbor_idxs(II, j, approx_order)[1:2]]) / s.dxs[j]
    forward_finite_difference(II, j, u) = dot(forward_weights(II, j), s.discvars[u][central_neighbor_idxs(II, j, approx_order)[2:3]]) / s.dxs[j]
    # TODO: improve interpolation of g(x) = u(x) for calculating u(x+-dx/2)
    interpolate_discrete_depvar(II, s, x, u, l) = sum([s.discvars[u][central_neighbor_idxs(II, s.x2i[x], approx_order)][i] for i in (l == 1 ? [2,3] : [1,2])]) / 2.
    # iv_mid returns middle space values. E.g. x(i-1/2) or y(i+1/2).
    interpolate_discrete_indvar(II, j, l) = (s.grid[j][II[j]] + s.grid[j][II[j]+l]) / 2.0
    # Dependent variable rules
    map_vars_to_discrete(II, x, l) = [u => interpolate_discrete_depvar(II, s, x, u, l) for u in s.vars]
    # Independent variable rules
    map_params_to_discrete(II, l) = [x => interpolate_discrete_indvar(II, x, l) for x in s.nottime]
    # Replacement rules: new approach
    # Calc
    function discrete_cartesian(expr, x, u)
        u_half_down = Num(substitute(substitute(expr, map_vars_to_discrete(II, x, -1)), map_params_to_discrete(II, -1)))
        u_half_up = Num(substitute(substitute(expr, map_vars_to_discrete(II, x, 1)), map_params_to_discrete(II, 1)))
        return dot([disc_downwind, disc_upwind], 
                   [reverse_finite_difference(II, x, u), forward_finite_difference(II, x, u)])
    end

    cartesian_deriv_rules = [@rule ($(Differential(iv))(*(~~a, $(Differential(iv))(dv), ~~b))) => discrete_cartesian(*(a..., b...),j,k) for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]

    cartesian_deriv_rules = vcat(vec(cartesian_deriv_rules),vec(
                            [@rule ($(Differential(iv))($(Differential(iv))(dv)/~a)) =>
                            discrete_cartesian(1/~a,j,k)
                            for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]))

    spherical_deriv_rules = [@rule *(~~a, ($(Differential(iv))((iv^2)*$(Differential(iv))(dv))), ~~b) / (iv^2) =>
                                *(~a..., central_deriv_spherical(II, j, k), ~b...)
                                for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]

    # r^-2 needs to be handled separately
    spherical_deriv_rules = vcat(vec(spherical_deriv_rules),vec(
                            [@rule *(~~a, (iv^-2) * ($(Differential(iv))((iv^2)*$(Differential(iv))(dv))), ~~b) =>
                                *(~a..., central_deriv_spherical(II, j, k), ~b...)
                                for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]))

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