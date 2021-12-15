function calculate_weights_cartesian(order::Int, x0::T, xs::AbstractVector, idxs::AbstractVector) where T<:Real
    # Cartesian domain: use Fornberg
    calculate_weights(order, x0, vec(xs[idxs]))
end
function calculate_weights_spherical(order::Int, x0::T, x::AbstractVector, idxs::AbstractVector) where T<:Real
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

function generate_finite_difference_rules(II, s, pde, discretization)
    approx_order = discretization.centered_order
    I1 = oneunit(first(s.Igrid))

    Imin(I1, order) = first(s.Igrid) + I1 * (order ÷ 2)
    Imax(I1, order) = last(s.Igrid) - I1 * (order ÷ 2)

    #--------------------------------------------------
    # * The stencil is the tappoints of the finite difference operator, relative to the current index
    stencil(j, order) = CartesianIndices(Tuple(map(x -> -x:x, (1:length(s.nottime) .== j) * (order ÷ 2))))
    
    # TODO: Generalize central difference handling to allow higher even order derivatives
    # The central neighbour indices should add the stencil to II, unless II is too close
    # to an edge in which case we need to shift away from the edge
    
    central_neighbor_idxs(II, j, order) = stencil(j, order) .+ max(Imin(I1, order), min(II, Imax(I1, order)))
    central_weights_cartesian(d_order, II, j) = calculate_weights_cartesian(d_order, s.grid[j][II[j]], s.grid[j], vec(map(i -> i[j],
        central_neighbor_idxs(II, j, approx_order))))
    central_deriv(d_order, II, j, k) = dot(central_weights(d_order, II, j), s.discvars[k][central_neighbor_idxs(II, j, approx_order)])

    central_deriv_cartesian(d_order, II, j, k) = dot(central_weights_cartesian(d_order, II, j), s.discvars[k][central_neighbor_idxs(II, j, approx_order)])

    # spherical Laplacian has a hardcoded order of 2 (only 2nd order is implemented)
    # both for derivative order and discretization order
    central_weights_spherical(II, j) = calculate_weights_spherical(2, s.grid[j][II[j]], s.grid[j], vec(map(i -> i[j], central_neighbor_idxs(II, j, 2))))
    central_deriv_spherical(II, j, k) = dot(central_weights_spherical(II, j), s.discvars[k][central_neighbor_idxs(II, j, 2)])

    # get a sorted list derivative order such that highest order is first. This is useful when substituting rules
    # starting from highest to lowest order.
    d_orders(x) = reverse(sort(collect(union(differential_order(pde.rhs, x), differential_order(pde.lhs, x)))))

    # orders = [d_orders(s) for s in s.nottime]
    # D = [CenteredDifference{1} for i in 1:length(s.nottime), d in dorders()]

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]
    central_deriv_rules_cartesian = Array{Pair{Num,Num},1}()
    for (j, x) in enumerate(s.nottime)
        rs = [(Differential(x)^d)(u) => central_deriv_cartesian(d, II, j, k) for d in d_orders(x), (k, u) in enumerate(s.vars)]
        for rule in rs
            push!(central_deriv_rules_cartesian, rule)
        end
    end

    central_deriv_rules_spherical = [Differential(r)(r^2 * Differential(r)(u)) / r^2 => central_deriv_spherical(II, j, k)
                                     for (j, r) in enumerate(s.nottime), (k, u) in enumerate(s.vars)]

    valrules = map_symbolic_to_discrete(II, s)

    # ! Use DerivativeOperator to get the coefficients, write a new constructor
    # TODO: upwind rules needs interpolation into `@rule`
    # upwinding_rules = [@rule(*(~~a,$(Differential(s.nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
    #                         *(~~a..., ~~b..., dot(reverse_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[1:2]])),
    #                         *(~~a..., ~~b..., dot(forward_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[2:3]]))))
    #                         for j in 1:length(s.nottime), k in 1:length(pdesys.s.vars)]


    return vcat(nonlinear_laplacian_rules(II, I1, pde, s, discretization, central_deriv_spherical),
        vec(central_deriv_rules_cartesian),
        vec(central_deriv_rules_spherical),
        valrules)
end

function nonlinear_laplacian_rules(II, I1, pde, s, discretization, central_deriv_spherical)
    approx_order = discretization.centered_order
    # # Discretization of non-linear laplacian.
    # d/dx( a du/dx ) ~ (a(x+1/2) * (u[i+1] - u[i]) - a(x-1/2) * (u[i] - u[i-1]) / dx^2
    stencil(j, order) = CartesianIndices(Tuple(map(x -> -x:x, (1:length(s.nottime) .== j) * (order ÷ 2))))
    Imin(I1, order) = first(s.Igrid) + I1 * (order ÷ 2)
    Imax(I1, order) = last(s.Igrid) - I1 * (order ÷ 2)
    
    central_neighbor_idxs(II, j, order) = stencil(j, order) .+ max(Imin(I1, order), min(II, Imax(I1, order)))
    # TODO: upwind rules needs interpolation into `@rule`


    forward_weights(II, j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j], II[j] + 1]])
    reverse_weights(II, j) = -calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j] - 1, II[j]]])
    

    reverse_finite_difference(II, j, k) = -dot(reverse_weights(II, j), s.discvars[k][central_neighbor_idxs(II, j, approx_order)[1:2]]) / s.dxs[j]
    forward_finite_difference(II, j, k) = dot(forward_weights(II, j), s.discvars[k][central_neighbor_idxs(II, j, approx_order)[2:3]]) / s.dxs[j]

    # TODO: improve interpolation of g(x) = u(x) for calculating u(x+-dx/2)
    interpolate_discrete_depvar(II, j, k, l) = sum([s.discvars[k][central_neighbor_idxs(II, j, approx_order)][i] for i in (l == 1 ? [2, 3] : [1, 2])]) / 2.0
    # interpolate_discrete_indvar returns middle space values. E.g. x(i-1/2) or y(i+1/2).
    interpolate_discrete_indvar(II, j, l) = (s.grid[j][II[j]] + s.grid[j][II[j]+l]) / 2.0
    # Dependent variable rules
    map_depvars_to_discrete(II, j, l) = [s.vars[k] => interpolate_discrete_depvar(II, j, k, l) for k = 1:length(s.vars)]
    # Independent variable rules
    map_indvars_to_discrete(II, l) = [s.nottime[j] => interpolate_discrete_indvar(II, j, l) for j = 1:length(s.nottime)]

    build_discrete_symbolic_expression(II, expr, j, l) = Num(substitute(substitute(expr, map_depvars_to_discrete(II, j, l)), map_indvars_to_discrete(II, l)))

    function cartesian_deriv(expr, II, j, k)
        discrete_expression_upwind = build_discrete_symbolic_expression(II, expr, j, 1)
        discrete_expression_downwind = build_discrete_symbolic_expression(II, expr, j, -1)
        # We need to do 
        discrete_expression_combined = [discrete_expression_downwind, discrete_expression_upwind]
        combined_finite_difference = [reverse_finite_difference(II, j, k), forward_finite_difference(II, j, k)]
        return dot(discrete_expression_combined, combined_finite_difference)
    end

    function generate_cartesian_deriv_rules()
        rules = [@rule ($(Differential(iv))(*(~~a, $(Differential(iv))(dv), ~~b))) =>
            cartesian_deriv(*(~~a..., ~~b...), II, j, k) for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]

        rules = vcat(vec(rules), vec(
            [@rule ($(Differential(iv))($(Differential(iv))(dv) / ~a)) =>
            cartesian_deriv(1 / ~a, II, j, k) for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]))

        return rules
    end

    function generate_spherical_deriv_rules()
        rules = [@rule *(~~a, ($(Differential(iv))((iv^2) * $(Differential(iv))(dv))), ~~b) / (iv^2) =>
            *(~a..., central_deriv_spherical(II, j, k), ~~b...)
                 for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]# r^-2 needs to be handled separately
        rules = vcat(vec(rules), vec(
            [@rule *(~~a, (iv^-2) * ($(Differential(iv))((iv^2) * $(Differential(iv))(dv))), ~~b) =>
            *(~a..., central_deriv_spherical(II, j, k), ~~b...) for (j, iv) in enumerate(s.nottime) for (k, dv) in enumerate(s.vars)]))
        return rules
    end

    cartesian_deriv_rules = vec(generate_cartesian_deriv_rules())
    spherical_deriv_rules = vec(generate_spherical_deriv_rules())

    # If lhs or rhs is a linear combination, apply rules term by term
    rhs_arg = istree(pde.rhs) && (SymbolicUtils.operation(pde.rhs) == +) ? SymbolicUtils.arguments(pde.rhs) : [pde.rhs]
    lhs_arg = istree(pde.lhs) && (SymbolicUtils.operation(pde.lhs) == +) ? SymbolicUtils.arguments(pde.lhs) : [pde.lhs]

    nonlinlap_rules = []

    for term in vcat(lhs_arg, rhs_arg)
        for rule in vcat(cartesian_deriv_rules, spherical_deriv_rules)
            if rule(term) !== nothing
                push!(nonlinlap_rules, term => rule(term))
            end
        end
    end


    return vec(nonlinlap_rules)
end