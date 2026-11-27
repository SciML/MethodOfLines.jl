@inline function clip_interior!!(lower, upper, s, b::PDEBase.AbstractBoundary)
    # This x2i is correct
    dim = x2i(s, depvar(b.u, s), b.x)
    @assert dim !== nothing "Internal Error: Variable $(b.x) not found in $(depvar(b.u, s)), when parsing boundary condition $(b)"
    if b isa InterfaceBoundary && isupper(b)
        return
    end
    lower[dim] = lower[dim] + !isupper(b)
    return upper[dim] = upper[dim] + isupper(b)
end

struct InteriorMap <: AbstractVarEqMapping
    var::Any
    pde::Any
    I::Any
    lower::Any
    upper::Any
    stencil_extents::Any
end

PDEBase.get_eqvar(im::InteriorMap, pde) = im.var[pde]
#to get an equal mapping, you want to associate every equation to a unique dependent variable that it's solving for
# this is tearing
# so there's u and v
# first equation has second derivative of u so we assign that one to u
# second equation, assume it has second derivative of both
# then we assign v to it because u is already assigned somewhere else.
# and use the interior based on the assignment

# Mismatched-grid interfaces are only coordinate-aware on the first-order (advection)
# path via bcoord; centered/half-offset paths are not. Reject orders > 1 up front.
function validate_interface_orders(pdes, boundarymap, discretization)
    for b in filter_interfaces(flatten_vardict(boundarymap))
        mismatched_interface_dxs(b, discretization) || continue
        orders = union(d_orders(b.x, pdes), d_orders(b.x2, pdes))
        higher = filter(>(1), orders)
        if !isempty(higher)
            throw(
                ArgumentError(
                    "Interface $(b.eq) connects $(b.x) and $(b.x2) with mismatched nonuniform grids, " *
                        "but the system contains spatial derivative orders $(sort(higher)) in these variables. " *
                        "Only first-order (advection) derivatives are supported across mismatched grids: " *
                        "higher-order stencils are not coordinate-aware at the interface and would be " *
                        "silently inaccurate. Use identical step sizes for the connected variables, or " *
                        "post an issue on GitHub if you need this feature."
                )
            )
        end
    end
    return
end

# The (at most `n`) nodes of `grid` next to the seam of an interface, ordered away from the seam.
function _seam_nodes(grid, n, atupper)
    m = min(n, length(grid) - 1)
    return atupper ? [grid[end - i] for i in 0:m] : [grid[1 + i] for i in 0:m]
end

# Gaps of equal nominal steps differ by at most one unit in the last place of the node coordinates
# (nodes are correctly rounded), and the rounded endpoints add at most half of one more. The gaps
# must also agree to sqrt(eps) of a gap, which is where the rounding bound alone stops separating
# rounding from a real difference (coordinates a few gaps from overflowing the significand).
function _gaps_match(gaps, nodes)
    lo, hi = extrema(gaps)
    T = float(typeof(hi))
    return hi - lo <= min(3 * eps(float(maximum(abs, nodes))) / 2, sqrt(eps(T)) * hi)
end

function _mentions(term, ops)
    iscall(term) || return false
    return any(op -> isequal(operation(term), op), ops) || any(a -> _mentions(a, ops), arguments(term))
end

# Variables of the `Differential`s inside `term`.
function differential_variables(term)
    iscall(term) || return []
    inner = reduce(vcat, differential_variables.(arguments(term)); init = [])
    return operation(term) isa Differential ? push!(inner, operation(term).x) : inner
end

# Orders in `x` of the derivatives whose argument holds one of the connected variables `ops`; a
# mixed derivative such as Dx(Dy(u)) is `:mixed`, which d_orders reports as first order in x
# although its stencil is centered in x. Derivatives of other fields in the same equation have no
# stencil across the interface.
function connected_derivative_orders(x, term, ops)
    term = safe_unwrap(term)
    iscall(term) || return []
    op = operation(term)
    if op isa Differential && _mentions(term, ops)
        inner = differential_variables(term)
        mixed = any(w -> !isequal(w, x), inner) && any(w -> isequal(w, x), inner)
        return Any[mixed ? :mixed : PDEBase.count_differentials(term, x)]
    end
    return reduce(vcat, (connected_derivative_orders(x, a, ops) for a in arguments(term)); init = [])
end

# What, in the equations `pdes`, discretizes with a stencil that is not coordinate-aware across a
# seam with unequal gaps; `nothing` when only first-order `UpwindScheme()` derivatives occur.
# Node vectors on both sides keep their coordinate-aware first-order schemes (nonuniform upwind and WENO).
function seam_unsafe_derivative(pdes, xs, ops, scheme, vectors)
    sides = [side for pde in pdes for side in (pde.lhs, pde.rhs)]
    found = [o for x in safe_unwrap.(xs) for side in sides for o in connected_derivative_orders(x, side, ops)]
    :mixed in found && return "a mixed derivative"
    orders = sort(unique(filter(>(0), found)))
    higher = filter(>(1), orders)
    isempty(higher) || return "derivative order $(join(higher, ", "))"
    isempty(orders) && return nothing
    (vectors || (scheme isa UpwindScheme && scheme.order == 1)) && return nothing
    label = scheme isa UpwindScheme ? "UpwindScheme($(scheme.order))" :
        scheme isa FunctionalScheme ? "$(scheme.name) advection" : string(typeof(scheme))
    return "first-order derivatives discretized with $label"
end

validate_interface_seams(pdes, boundarymap, s, discretization) = nothing

# Connected axes may have unequal grid spacings at the seam only for first-order `UpwindScheme()`,
# whose stencils use the actual coordinates (src/discretization/schemes/upwind_difference). Every
# other stencil that straddles the seam would use one spacing and be silently wrong.
function validate_interface_seams(
        pdes, boundarymap, s, discretization::MOLFiniteDifference
    )
    for b in filter_interfaces(flatten_vardict(boundarymap))
        isequal(b.x, b.x2) && continue
        mismatched_interface_dxs(b, discretization) && continue
        n = discretization.approx_order
        nodes1 = _seam_nodes(s.grid[b.x], n, isupper(b))
        nodes2 = _seam_nodes(s.grid[b.x2], n, !isupper(b))
        gaps1 = abs.(diff(nodes1))
        gaps2 = abs.(diff(nodes2))
        _gaps_match(vcat(gaps1, gaps2), vcat(nodes1, nodes2)) && continue
        ops = [operation(safe_unwrap(b.u)), operation(safe_unwrap(b.u2))]
        vectors = !(s.grid[b.x] isa StepRangeLen) && !(s.grid[b.x2] isa StepRangeLen)
        what = seam_unsafe_derivative(
            pdes, (b.x, b.x2), ops, discretization.advection_scheme, vectors
        )
        what === nothing && continue
        throw(
            ArgumentError(
                "Interface $(b.eq) connects $(b.x) and $(b.x2), whose grid gaps next to the interface " *
                    "differ (nearest first: $gaps1 in $(b.x), $gaps2 in $(b.x2)), but the system contains " *
                    "$what in these variables. Only first-order `UpwindScheme()` derivatives are " *
                    "coordinate-aware across such an interface; the other stencils would use a single " *
                    "spacing and be silently inaccurate. Use equal gaps on both sides of the interface " *
                    "(for example equal step sizes on intervals of equal length)."
            )
        )
    end
    return
end

function PDEBase.construct_var_equation_mapping(
        pdes::Vector{Equation}, boundarymap, s::DiscreteSpace{N, M},
        discretization::MOLDiscretization
    ) where {N, M}
    @assert length(pdes) == M "There must be the same number of equations and unknowns, got $(length(pdes)) equations and $(M) unknowns"
    validate_interface_orders(pdes, boundarymap, discretization)
    validate_interface_seams(pdes, boundarymap, s, discretization)
    m = buildmatrix(pdes, s)
    varmap = Dict(build_variable_mapping(m, s.ū, pdes))

    # Determine the interiors for each pde
    vlower = []
    vupper = []
    extents = []

    interior = map(pdes) do pde
        u = varmap[pde]
        boundaries = mapreduce(x -> boundarymap[operation(u)][x], vcat, s.x̄)
        n = ndims(u, s)
        lower = zeros(Int, n)
        upper = zeros(Int, n)
        # Determine thec number of points to remove from each end of the domain for each dimension
        for b in boundaries
            #@show b
            clip_interior!!(lower, upper, s, b, discretization)
        end
        push!(vlower, pde => lower)
        push!(vupper, pde => upper)
        #TODO: Allow asymmetry
        pdeorders = Dict(map(x -> x => d_orders(x, [pde]), s.x̄))

        # Add ghost points to pad stencil extents
        lowerextents,
            upperextents = calculate_stencil_extents(
            s, u, discretization, pdeorders, boundarymap
        )
        push!(extents, pde => (lowerextents, upperextents))
        lower = [max(e, l) for (e, l) in zip(lowerextents, lower)]
        upper = [max(e, u) for (e, u) in zip(upperextents, upper)]
        mindomsize = lower .+ upper .+ 1
        if any(tup -> mindomsize[tup[1]] > length(s, tup[2]), enumerate(ivs(u, s)))
            error("The domain is too small to support the requested discretization, got domain size of $(size(s)).")
        end
        validate_boundary_stencil_length(pde, u, s, pdeorders, boundarymap, discretization)
        # Don't update this x2i, it is correct.
        pde => generate_interior(lower, upper, u, s, discretization)
    end

    pdemap = [k.second => k.first for k in varmap]
    return InteriorMap(
        varmap, Dict(pdemap), Dict(interior), Dict(vlower), Dict(vupper), Dict(extents)
    )
end

validate_boundary_stencil_length(pde, u, s, pdeorders, boundarymap, discretization) = nothing

# An axis end that is not periodic or an interface is closed with one-sided stencils, which
# take derivative order + approx_order points along the axis.
function validate_boundary_stencil_length(
        pde, u, s, pdeorders, boundarymap, discretization::MOLFiniteDifference
    )
    p = discretization.approx_order
    for x in ivs(u, s)
        isempty(pdeorders[x]) && continue
        haslower, hasupper = haslowerupper(filter_interfaces(boundarymap[operation(u)][x]), x)
        haslower && hasupper && continue
        d = maximum(pdeorders[x])
        n = length(s, x)
        n >= d + p && continue
        throw(
            ArgumentError(
                "$pde has an order $d derivative in $x. At approx_order = $p its one-sided " *
                    "boundary stencil needs at least $(d + p) grid points in $x, but there are $n. " *
                    "Use a finer grid in $x or a lower approx_order."
            )
        )
    end
    return
end

function generate_interior(lower, upper, u, s, disc::MOLDiscretization)
    args = remove(arguments(u), s.time)

    ret = s.Igrid[u][
        [
            ((1 + lower[x2i(s, u, x)]):(length(s.grid[x]) - upper[x2i(s, u, x)]))
                for x in args
        ]...,
    ]
    return ret
end

function calculate_stencil_extents(s, u, discretization, orders, bcmap)
    aorder = discretization.approx_order
    advection_scheme = discretization.advection_scheme

    args = remove(arguments(u), s.time)
    lowerextents = zeros(Int, length(args))
    upperextents = zeros(Int, length(args))

    for (j, x) in enumerate(args)
        # Skip if periodic in x
        haslower, hasupper = haslowerupper(filter_interfaces(bcmap[operation(u)][x]), x)
        for dorder in filter(isodd, orders[x])
            ascheme = dorder == 1 ? advection_scheme : UpwindScheme()
            if !haslower
                lowerextents[j] = max(lowerextents[j], extent(ascheme, dorder, s.dxs[x]))
            end
            if !hasupper
                upperextents[j] = max(upperextents[j], extent(ascheme, dorder, s.dxs[x]))
            end
        end
    end
    return lowerextents, upperextents
end

function buildmatrix(pdes, s::DiscreteSpace{N, M}) where {N, M}
    m = zeros(Int, M, M)
    elegiblevars = [getvarmap(pde, s) for pde in pdes]
    u2i = Dict([u => k for (k, u) in enumerate(s.ū)])
    #@show elegiblevars, s.ū
    for (i, varmap) in enumerate(elegiblevars)
        for var in keys(varmap)
            m[i, u2i[var]] = varmap[var]
        end
    end
    return m
end

function build_variable_mapping(m, vars, pdes)
    notzero(x) = x > 0 ? 1 : 0
    varpdemap = []
    N = length(pdes)
    rows = sum(m, dims = 2)
    cols = sum(m, dims = 1)
    i = findfirst(isequal(0), rows)
    j = findfirst(isequal(0), cols)
    @assert i === nothing "Equation $(pdes[i[1]]) is not an equation for any of the dependent variables."
    @assert j === nothing "Variable $(vars[j[2]]) does not appear in any equation, therefore cannot be solved for"
    for k in 1:N
        # Check if any of the pdes only have one valid variable
        m_ones = notzero.(m)
        cols = sum(m_ones, dims = 1)
        j = findfirst(isequal(1), cols)
        if j !== nothing
            j = j[2]
            for i in 1:N
                if m[i, j] > 0
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] .= 0
                    m[:, j] .= 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables only have one valid pde
        rows = sum(m_ones, dims = 2)
        i = findfirst(isequal(1), rows)
        if i !== nothing
            i = i[1]
            for j in 1:N
                if m[i, j] > 0
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] .= 0
                    m[:, j] .= 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables have more than one valid pde, and pick one
        I = findmax(m)[2]
        i = I[1]
        j = I[2]
        push!(varpdemap, pdes[i] => vars[j])
        m[i, :] .= 0
        m[:, j] .= 0
    end
    @assert length(varpdemap) == N "Could not map all PDEs to variables to solve for, the system is unbalanced."
    return varpdemap
end

@inline function split(children)
    count = [child[1] for child in children]
    # if there are more than one depvars at that order, include all of them
    is = findall(a -> a == maximum(count), count)
    vars = unique(reduce(vcat, [child[2] for child in children[is]]))
    return (sum(count), vars)
end
"""
Creates a ranking of the variables in the term, based on their derivative order.
The heuristic that should work is, if there's a time derivative then use that variable, otherwise use the highest derivative for that variable. If there are two with the highest derivative, pick first from the list that hasn't been chosen for another equation
"""
function get_ranking!(varmap, term, x, s)
    if !iscall(term)
        return (0, [])
    end
    S = Symbolics
    SU = SymbolicUtils
    #@show term
    if findfirst(isequal(term), s.ū) !== nothing
        if varmap[term] < 1
            varmap[term] = 1
        end
        return (1, [term])
    else
        op = SU.operation(term)
        children = map(arg -> get_ranking!(varmap, arg, x, s), SU.arguments(term))
        count, vars = split(children)
        if op isa Differential && isequal(op.x, x)
            for var in vars
                if varmap[var] < count + op.order
                    varmap[var] = count + op.order
                end
            end
            return (op.order + count, vars)
        end
        return (count, vars)
    end
end

function getvarmap(pde, s)
    ct = 0
    ut = []
    # Create ranking for each variable
    varmap = Dict([u => 0 for u in s.ū])
    if s.time !== nothing
        l = get_ranking!(varmap, pde.lhs, s.time, s)
        r = get_ranking!(varmap, pde.rhs, s.time, s)
        for u in s.ū
            if varmap[u] > 1
                varmap[u] += div(typemax(Int), 2) #Massively weight derivatives in time
            end
        end
    end
    for x in s.x̄
        l = get_ranking!(varmap, pde.lhs, x, s)
        r = get_ranking!(varmap, pde.rhs, x, s)
    end
    return varmap
end
