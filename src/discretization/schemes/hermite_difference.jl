@inline function _hermite_operation(a)
    av = unwrap(a)
    return iscall(av) ? operation(av) : av
end

@inline function _same_hermite_variable(a, b)::Bool
    return isequal(_hermite_operation(a), _hermite_operation(b))
end

@inline function _resolve_hermite_variable(v, s)
    haskey(s.discvars, v) && return v
    for candidate in keys(s.discvars)
        _same_hermite_variable(candidate, v) && return candidate
    end
    return nothing
end

_hermite_is_diffusive(scheme::HermiteScheme, u, x) =
    any(k -> _same_hermite_variable(k[1], u) && isequal(k[2], x), scheme.diffusive)

@inline function _hermite_partner(scheme::HermiteScheme, u, x, s)
    for (key, value) in scheme.deriv_var
        if key isa Tuple && length(key) == 2
            _same_hermite_variable(key[1], u) && isequal(key[2], x) || continue
            return _resolve_hermite_variable(value, s)
        elseif _same_hermite_variable(key, u)
            length(ivs(u, s)) == 1 || throw(
                ArgumentError(
                    "HermiteScheme mapping $(key) => $(value) is ambiguous for " *
                        "multidimensional variable $(u). Use direction-specific mappings " *
                        "such as (u, x) => vx and (u, y) => vy."
                )
            )
            return _resolve_hermite_variable(value, s)
        end
    end
    return nothing
end

@inline function _hermite_weights_and_taps(D, H, II, s, bs, jx, u; wide = false)
    j, x = jx
    if !isempty(bs) && !(s.grid[x] isa StepRangeLen)
        throw(
            ArgumentError(
                "HermiteScheme does not support periodic or interface stencils " *
                    "on a nonuniform grid in $x."
            )
        )
    end
    _, Itap = central_difference_weights_and_stencil(D, II, s, bs, jx, u)

    if any(I -> I isa RefCartesianIndex, Itap)
        self_periodic = !isempty(bs) && all(bs) do b
            isequal(b.x, b.x2) && _same_hermite_variable(b.u, u) &&
                _same_hermite_variable(b.u2, u)
        end
        self_periodic || throw(
            ArgumentError(
                "HermiteScheme does not yet support cross-variable interface stencils."
            )
        )
        return H.d_weights, H.e_weights, [I isa RefCartesianIndex ? I.I : I for I in Itap]
    end

    interior = s.grid[x] isa StepRangeLen &&
        II[j] > D.boundary_point_count &&
        II[j] <= length(s, x) - D.boundary_point_count &&
        length(Itap) == H.stencil_length
    interior && return H.d_weights, H.e_weights, Itap
    Itap = _hermite_boundary_stencil(D, II, s, j, x, Itap; wide)
    coords = [s.grid[x][I[j]] for I in Itap]
    x0 = s.grid[x][II[j]]
    dweights, eweights = calculate_weights(D.derivative_order, x0, coords; dfdx = true)
    return dweights, eweights, Itap
end

"""
    _hermite_boundary_stencil(D, II, s, j, x, Itap; wide = false)

Grid points of the Hermite stencil for `D` near a boundary of `x`, in place of MOL's
one-sided points `Itap`. Hermite weights on one-sided points make the semi-discrete system
unstable above `approx_order = 2`, so the point `k` steps from the boundary uses the
centred window of `2k + 1` points, clipped to the grid, with between `d + 1` and
`length(Itap)` points. `wide = true`, for a PDE diffusive in `x`, raises the minimum to
`max(d + 1, cld(approx_order + d, 2) + 1)` points.
"""
function _hermite_boundary_stencil(D, II, s, j, x, Itap; wide = false)
    d = D.derivative_order
    m = wide ? max(d + 1, cld(D.approximation_order + d, 2) + 1) : d + 1
    n = length(s, x)
    i = II[j]
    bpc = D.boundary_point_count
    lower = i <= bpc
    ((lower || i > n - bpc) && (length(Itap) > d + 2 || d >= 3) && length(Itap) > m) ||
        return Itap
    k = lower ? i - 1 : n - i
    w = min(max(m, 2k + 1), length(Itap))
    first = clamp(lower ? i - cld(w - 1, 2) : i - fld(w - 1, 2), 1, n - w + 1)
    I0 = Itap[1]
    return [CartesianIndex(ntuple(k -> k == j ? l : I0[k], length(I0))) for l in first:(first + w - 1)]
end

@inline function hermite_central_difference(D, H, II, s, bs, jx, u, v; wide = false)
    dweights, eweights, Itap = _hermite_weights_and_taps(D, H, II, s, bs, jx, u; wide)
    u_vals = [s.discvars[u][I] for I in Itap]
    v_vals = [s.discvars[v][I] for I in Itap]
    return sym_dot(dweights, u_vals) + sym_dot(eweights, v_vals)
end

function hermite_partner_derivative_rule(derivweights, s, v_, II, x_)
    v = depvar(v_, s)
    p = _hermite_partner_second_derivative(derivweights, s, v, x_)
    p === nothing && return nothing
    rhs = hermite_central_difference(
        p.D, p.H, II, s, [], (x2i(s, p.w, x_), x_), p.w, v; wide = p.wide
    )
    return (Differential(x_)^1)(v_) => rhs
end

function array_hermite_partner_derivative(derivweights, s, v, x_, ranges, indexmap, II0)
    p = _hermite_partner_second_derivative(derivweights, s, v, x_)
    p === nothing && return nothing
    w = p.w
    j = x2i(s, w, x_)
    dw, ew, Itap = _hermite_weights_and_taps(p.D, p.H, II0, s, [], (j, x_), w; wide = p.wide)
    offsets = [I[j] - II0[j] for I in Itap]
    slices = vcat(
        [array_slice(w, s, ranges, indexmap; shiftx = x_, offset = o) for o in offsets],
        [array_slice(v, s, ranges, indexmap; shiftx = x_, offset = o) for o in offsets]
    )
    return array_stencil(vcat(collect(dw), collect(ew)), slices)
end

function hermite_boundary_derivative_rules(derivweights, s, boundary, II, indexmap, bcmap)
    rules = Pair[]
    derivweights isa DifferentialDiscretizer || return rules
    derivweights.advection_scheme isa HermiteScheme || return rules
    get_grid_type(s) <: CenterAlignedGrid || return rules
    u_, x_ = getvars(boundary)
    u = depvar(u_, s)
    II = newindex(u_, II, s, indexmap)
    ufunc(v, I, _) = s.discvars[v][I]
    for y in ivs(u, s)
        isequal(y, x_) && continue
        bs = filter_interfaces(get(get(bcmap, operation(u), Dict()), y, []))
        for d in get(derivweights.orders, y, Int[])
            D = get(derivweights.map, Differential(y)^d, nothing)
            D === nothing && continue
            rhs = central_difference(D, II, s, bs, (x2i(s, u, y), y), u, ufunc)
            push!(rules, (Differential(y)^d)(u_) => rhs)
        end
    end
    scheme = derivweights.advection_scheme
    if scheme.automatic && _hermite_partner(scheme, u, x_, s) !== nothing
        g = s.grid[x_]
        D = CompleteCenteredDifference(
            1, length(g) >= 14 ? 6 : length(g) >= 5 ? 4 : 2, g isa StepRangeLen ? s.dxs[x_] : g
        )
        push!(rules, Differential(x_)(u_) => central_difference(D, II, s, [], (x2i(s, u, x_), x_), u, ufunc))
    end
    return rules
end

function hermite_bc_fallback(derivweights, s, boundary, u, x_)
    derivweights isa DifferentialDiscretizer || return nothing
    derivweights.advection_scheme isa HermiteScheme || return nothing
    for y in ivs(u, s)
        isequal(y, x_) && continue
        any(side -> !isempty(_x_derivative_orders!(Set{Int}(), side, y)), (boundary.eq.lhs, boundary.eq.rhs)) &&
            throw(ArrayFormFallback("HermiteScheme boundary condition with a derivative in $y along the face"))
    end
    ex = Symbolics.value(boundary.eq.lhs - boundary.eq.rhs)
    any(w -> !isequal(depvar(w, s), u) && subsmatch(ex, Symbolics.value(Differential(x_)(w)) => nothing), boundary.depvars) &&
        throw(ArrayFormFallback("HermiteScheme boundary condition with a derivative of another variable"))
    return nothing
end

function _hermite_partner_second_derivative(derivweights, s, v, x)
    derivweights isa DifferentialDiscretizer || return nothing
    scheme = derivweights.advection_scheme
    scheme isa HermiteScheme || return nothing
    w = _hermite_primary(scheme, v, x, s)
    w === nothing && return nothing
    D2 = Differential(x)^2
    (haskey(derivweights.map, D2) && haskey(derivweights.hermitemap, D2)) || return nothing
    return (;
        w, D = derivweights.map[D2], H = derivweights.hermitemap[D2],
        wide = _hermite_is_diffusive(scheme, w, x),
    )
end

function _hermite_primary(scheme::HermiteScheme, v, x, s)
    for (key, value) in scheme.deriv_var
        _same_hermite_variable(value, v) || continue
        if key isa Tuple && length(key) == 2
            isequal(key[2], x) && return _resolve_hermite_variable(key[1], s)
        elseif length(ivs(v, s)) == 1
            return _resolve_hermite_variable(key, s)
        end
    end
    return nothing
end

"""
    generate_hermite_rules(II, s, depvars, derivweights, bcmap, indexmap, terms)

Rules for all spatial derivatives at index `II` when `advection_scheme isa HermiteScheme`,
in place of `generate_cartesian_rules`: Hermite weights for a variable with a derivative
partner in that direction, central differences otherwise.
"""
function generate_hermite_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms
    )
    scheme = derivweights.advection_scheme
    central_ufunc(u, I, _) = s.discvars[u][I]
    rules = Pair[]
    for u in depvars
        for x in ivs(u, s)
            partner = _hermite_partner(scheme, u, x, s)
            jx = (x2i(s, u, x), x)
            bs = filter_interfaces(bcmap[operation(u)][x])
            for d in derivweights.orders[x]
                lhs = (Differential(x)^d)(u)
                rhs = if partner !== nothing
                    hermite_central_difference(
                        derivweights.map[Differential(x)^d],
                        derivweights.hermitemap[Differential(x)^d],
                        Idx(II, s, u, indexmap), s, bs, jx, u, partner;
                        wide = _hermite_is_diffusive(scheme, u, x)
                    )
                else
                    central_difference(
                        derivweights.map[Differential(x)^d], Idx(II, s, u, indexmap),
                        s, bs, jx, u, central_ufunc
                    )
                end
                push!(rules, lhs => rhs)
            end
        end
    end
    append!(rules, hermite_mixed_derivative_rules(II, s, depvars, derivweights, bcmap, indexmap))
    return rules
end

"""
    hermite_mixed_derivative_rules(II, s, depvars, derivweights, bcmap, indexmap)

Rules for `Dx(Dy(u))`, `Dy(Dx(u))`, `Dy(v)` and `Dx(w)` of every `u` with partners
`v = u_x` and `w = u_y`, all as `u_xy ≈ δy v + δx w - δx δy u` with first-derivative
stencils `δx` and `δy`. The error is the product of the two stencil errors, twice the order
of `δx δy u`.
"""
function hermite_mixed_derivative_rules(II, s, depvars, derivweights, bcmap, indexmap)
    rules = Pair[]
    bs(var, iv) = filter_interfaces(bcmap[operation(var)][iv])
    tb(var, iv) = filter(b -> !(b isa InterfaceBoundary), get(bcmap[operation(var)], iv, []))
    dir(var, iv) = (x2i(s, var, iv), iv)
    for term in _hermite_mixed_derivative_terms(s, depvars, derivweights)
        u, v, w, x, y = term.u, term.v, term.w, term.x, term.y
        Dx1, Dy1 = derivweights.map[Differential(x)], derivweights.map[Differential(y)]
        Iv, Iw, Iu = (Idx(II, s, q, indexmap) for q in (v, w, u))
        wy, Ty = _mixed_first_difference(Dy1, Iv, s, bs(v, y), dir(v, y), v; tbcs = tb(v, y))
        dy_v = sym_dot(wy, [s.discvars[v][I] for I in Ty])
        wx, Tx = _mixed_first_difference(Dx1, Iw, s, bs(w, x), dir(w, x), w; tbcs = tb(w, x))
        dx_w = sym_dot(wx, [s.discvars[w][I] for I in Tx])
        ux, Ux = _mixed_first_difference(Dx1, Iu, s, bs(u, x), dir(u, x), u; tbcs = tb(u, x))
        uy, Uy = _mixed_first_difference(Dy1, Iu, s, bs(u, y), dir(u, y), u; tbcs = tb(u, y))
        dxdy_u = sum(
            ux[a] * uy[b] * s.discvars[u][Iu + (Ux[a] - Iu) + (Uy[b] - Iu)]
                for a in eachindex(Ux), b in eachindex(Uy)
        )
        rhs = dy_v + dx_w - dxdy_u
        face = _hermite_face_mixed_derivative(II, s, u, v, w, x, y, bcmap, indexmap)
        face === nothing &&
            (face = _hermite_face_mixed_derivative(II, s, u, w, v, y, x, bcmap, indexmap))
        face === nothing || (rhs = face)
        for key in mixed_derivative_keys(u, x, 1, y, 1)
            push!(rules, key => rhs)
        end
        push!(rules, (Differential(y)^1)(v) => rhs)
        push!(rules, (Differential(x)^1)(w) => rhs)
    end
    return rules
end

function _mixed_first_difference(D, II, s, bs, jx, u; tbcs = [])
    j, x = jx
    weights, Itap = central_difference_weights_and_stencil(D, II, s, bs, jx, u)
    any(I -> I isa RefCartesianIndex, Itap) && return weights, Itap
    s.grid[x] isa StepRangeLen || return weights, Itap
    haslower, hasupper = haslowerupper(bs, x)
    n = length(s, x)
    low = II[j] <= D.boundary_point_count
    near = (low && !haslower) || (II[j] > n - D.boundary_point_count && !hasupper)
    near || return weights, Itap
    any(b -> low ? b isa LowerBoundary : b isa UpperBoundary, tbcs) || return weights, Itap
    taps = _hermite_boundary_stencil(D, II, s, j, x, Itap)
    length(taps) == length(Itap) && taps == Itap && return weights, Itap
    coords = [s.grid[x][I[j]] for I in taps]
    return calculate_weights(1, s.grid[x][II[j]], coords), taps
end

function _hermite_face_mixed_derivative(II, s, u, v, w, x, y, bcmap, indexmap)
    j = x2i(s, v, x)
    Iv = Idx(II, s, v, indexmap)
    n = length(s, x)
    Iv[j] == 1 || Iv[j] == n || return nothing
    onside(b) = Iv[j] == 1 ? b isa LowerBoundary : b isa UpperBoundary
    vbcs = filter(b -> onside(b) && b.order == 0, get(bcmap[operation(v)], x, []))
    length(vbcs) == 1 || return nothing
    any(onside, get(bcmap[operation(w)], x, [])) && return nothing
    b = only(vbcs)
    F = symbolic_linear_solve(b.eq, b.u)
    isequal(Symbolics.value(F), NaN) && return nothing
    Fy = expand_derivatives(Differential(y)(F))
    ub = operation(unwrap(u))(arguments(unwrap(b.u))...)
    Iu, Iw = Idx(II, s, u, indexmap), Idx(II, s, w, indexmap)
    ex = substitute(
        Fy, Dict(Differential(y)(ub) => s.discvars[w][Iw], ub => s.discvars[u][Iu])
    )
    ex = substitute(ex, Dict(y => s.grid[y][Iu[x2i(s, u, y)]]))
    _is_face_data(unwrap(ex), s, y) || return nothing
    return ex
end

function _is_face_data(ex, s, y)
    iscall(ex) || return !isequal(ex, unwrap(y))
    op = operation(ex)
    op isa Differential && return false
    if op === getindex
        return true
    end
    any(isequal(op, operation(unwrap(d))) for d in s.ū) && return false
    return all(a -> _is_face_data(unwrap(a), s, y), arguments(ex))
end

function array_validate_hermite(s, args, periodic)
    for x in args
        s.grid[x] isa StepRangeLen ||
            throw(ArrayFormFallback("HermiteScheme on a nonuniform grid in $x"))
    end
    for spec in values(periodic)
        array_wrap_is_twodomain(spec) &&
            throw(ArrayFormFallback("HermiteScheme across a two-domain interface"))
    end
    return nothing
end

function _hermite_mixed_derivative_terms(s, depvars, derivweights)
    scheme = derivweights.advection_scheme
    found = []
    for u in depvars
        xs = ivs(u, s)
        for x in xs, y in remove(xs, unwrap(x))
            v = _hermite_partner(scheme, u, x, s)
            w = _hermite_partner(scheme, u, y, s)
            (v === nothing || w === nothing) && continue
            keys = vcat(
                collect(mixed_derivative_keys(u, x, 1, y, 1)),
                [safe_unwrap((Differential(y)^1)(v)), safe_unwrap((Differential(x)^1)(w))]
            )
            push!(found, (; u, v, w, x, y, keys))
        end
    end
    return found
end

_hermite_pde_has(pde, key) = subsmatch(pde.lhs, key => nothing) || subsmatch(pde.rhs, key => nothing)

function array_hermite_mixed_orders!(mixedorders, pde, s, depvars, derivweights)
    for p in _hermite_mixed_derivative_terms(s, depvars, derivweights)
        any(k -> _hermite_pde_has(pde, k), p.keys) || continue
        push!(get!(Set{Int}, mixedorders, p.x), 1)
        push!(get!(Set{Int}, mixedorders, p.y), 1)
    end
    return mixedorders
end

"""
    array_hermite_rules(pde, s, depvars, pdeorders, derivweights, ranges, indexmap, periodic)

Rules of the Hermite array form in the core region, as `(mixedrules, derivrules)`: the
interior stencils of `generate_hermite_rules` and `hermite_mixed_derivative_rules` applied
to shifted slices. Throws `ArrayFormFallback` if an interior Hermite weight set is missing.
"""
function array_hermite_rules(pde, s, depvars, pdeorders, derivweights, ranges, indexmap, periodic)
    scheme = derivweights.advection_scheme
    derivrules = Pair[]
    for u in depvars, x in ivs(depvar(u, s), s)
        haskey(pdeorders, x) || continue
        partner = _hermite_partner(scheme, u, x, s)
        for d in pdeorders[x]
            D = Differential(x)^d
            Dop = derivweights.map[D]
            expr = if partner === nothing
                array_central_difference(Dop, s, u, x, d, ranges, indexmap, periodic)
            else
                H = get(derivweights.hermitemap, D, nothing)
                (H === nothing || H.d_weights === nothing) &&
                    throw(ArrayFormFallback("no interior Hermite weights for $D"))
                taps = half_range(H.stencil_length)
                sl(q) = [
                    array_slice(q, s, ranges, indexmap; shiftx = x, offset = k, periodic = periodic)
                        for k in taps
                ]
                array_stencil(vcat(collect(H.d_weights), collect(H.e_weights)), vcat(sl(u), sl(partner)))
            end
            push!(derivrules, safe_unwrap(D(u)) => expr)
        end
    end
    mixedrules = Pair[]
    for p in _hermite_mixed_derivative_terms(s, depvars, derivweights)
        any(k -> _hermite_pde_has(pde, k), p.keys) || continue
        Dx1, Dy1 = derivweights.map[Differential(p.x)], derivweights.map[Differential(p.y)]
        dy_v = array_central_difference(Dy1, s, p.v, p.y, 1, ranges, indexmap, periodic)
        dx_w = array_central_difference(Dx1, s, p.w, p.x, 1, ranges, indexmap, periodic)
        dxdy_u = array_mixed_difference(Dx1, Dy1, s, p.u, p.x, p.y, ranges, indexmap, periodic)
        rhs = broadcast(-, broadcast(+, dy_v, dx_w), dxdy_u)
        for key in p.keys
            push!(mixedrules, key => rhs)
        end
    end
    return mixedrules, derivrules
end

function array_hermite_corner_dirichlet(s, u, ranges, J, bcmap)
    args = ivs(u, s)
    all(j -> length(ranges[j]) == 1, J) || return nothing
    uop = operation(u)
    val = nothing
    for j in J
        x = args[j]
        atlow = first(ranges[j]) == 1
        raw = get(bcmap[uop], x, [])
        bcs = filter(b -> !(b isa InterfaceBoundary), raw)
        isempty(bcs) && return nothing
        length(bcs) == length(raw) || return nothing
        onside(b) = atlow ? b isa LowerBoundary : b isa UpperBoundary
        dir = filter(b -> onside(b) && b.order == 0, bcs)
        isempty(dir) && return nothing
        if val === nothing
            b = first(dir)
            F = symbolic_linear_solve(b.eq, b.u)
            isequal(Symbolics.value(F), NaN) && return nothing
            val = map(CartesianIndices(map(length, ranges))) do I
                sub = Dict(args[k] => s.grid[args[k]][ranges[k][I[k]]] for k in 1:length(args))
                substitute(expand_derivatives(F), sub)
            end
            all(w -> _is_corner_data(Symbolics.unwrap(w), s, args), val) || return nothing
        end
    end
    val === nothing && return nothing
    arr = array_variable(u, s)
    return arr[ranges...] ~ val
end

function _is_corner_data(ex, s, args)
    ex isa SymbolicUtils.BasicSymbolic || return true
    iscall(ex) || return !any(a -> isequal(ex, Symbolics.unwrap(a)), args)
    op = operation(ex)
    op isa Differential && return false
    op === getindex && return false
    any(isequal(op, operation(unwrap(d))) for d in s.ū) && return false
    return all(a -> _is_corner_data(a, s, args), arguments(ex))
end

const _fused_operations = (+, -, *, /)

"""
    FusedStencil{N, S, K, C}

One elementwise expression tree over a box of size `S`, evaluated by a compiled loop
(`kernel`) instead of nested broadcasts over slice copies. The loop applies the same
operations in the same order to the constants as given, and takes its element type from
the value at the first point, so the values and the element type are those of the
broadcasts. The node list is kept for scalarization.
"""
struct FusedStencil{N, S, K, C} <: Function
    kind::Vector{Symbol}
    operation_index::Vector{Int}
    children::Vector{Vector{Int}}
    source::Vector{Int}
    ranges::Vector{NTuple{N, UnitRange{Int}}}
    symbolic_value::Vector{Any}
    constant_arrays::C
    kernel::K
end
Base.size(::FusedStencil{N, S}) where {N, S} = S

function _fused_loop_expr(b, N, S, nargs)
    A = [Symbol("A", k) for k in 1:nargs]
    is = [Symbol("i", d) for d in 1:N]
    v(n) = Symbol("v", n)
    stmts = Expr[]
    for n in eachindex(b.kind)
        k = b.kind[n]
        rhs = if k == :constant
            b.symbolic_value[n]
        elseif k == :shifted
            ix = b.ranges[n]
            Expr(:ref, A[b.source[n]], [:($(first(ix[d]) - 1) + $(is[d])) for d in 1:N]...)
        elseif k == :array
            Expr(:ref, A[b.source[n]], is...)
        elseif k == :scalar
            A[b.source[n]]
        elseif k == :constant_array
            Expr(:ref, :(C[$(b.source[n])]), is...)
        else
            ks = b.children[n]
            f = GlobalRef(Base, nameof(_fused_operations[b.operation_index[n]]))
            if length(ks) == 1
                b.operation_index[n] == 2 ? Expr(:call, f, v(ks[1])) : v(ks[1])
            else
                acc = v(ks[1])
                for j in 2:length(ks)
                    acc = Expr(:call, f, acc, v(ks[j]))
                end
                acc
            end
        end
        push!(stmts, :($(v(n)) = $rhs))
    end
    first_point = Expr(
        :block, [:($(is[d]) = 1) for d in 1:N]..., stmts...,
        :(out = Array{typeof($(v(length(b.kind)))), $N}(undef, $S))
    )
    inner = Expr(:block, stmts..., :(out[$(is...)] = $(v(length(b.kind)))))
    loop = inner
    for d in 1:N
        loop = :(
            for $(is[d]) in 1:$(S[d])
                $loop
            end
        )
    end
    return :(
        function (C, $(A...))
            $first_point
            @inbounds $loop
            return out
        end
    )
end

function (op::FusedStencil{N, S})(As::Vararg{Any, K}) where {N, S, K}
    all(a -> isconcretetype(eltype(a)), (As..., op.constant_arrays...)) &&
        return op.kernel(op.constant_arrays, As...)
    return map(I -> _fused_node_symbolic(op, As, length(op.kind), I), CartesianIndices(S))
end

SymbolicUtils.promote_symtype(::FusedStencil{N}, ::Type...) where {N} = Array{Real, N}
function SymbolicUtils.promote_shape(::FusedStencil{N, S}, ::SymbolicUtils.ShapeT...) where {N, S}
    return SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, S))
end
SymbolicUtils.scalarization_function(::FusedStencil) = _fused_scalarize

function _fused_node_symbolic(op, As, n, I)
    N = length(I)
    k = op.kind[n]
    k == :constant && return op.symbolic_value[n]
    if k == :shifted
        ix = op.ranges[n]
        return unwrap_const(As[op.source[n]][ntuple(d -> ix[d][I[d]], N)...])
    end
    k == :array && return unwrap_const(As[op.source[n]][I])
    k == :scalar && return unwrap_const(As[op.source[n]])
    k == :constant_array && return op.constant_arrays[op.source[n]][I]
    args = [_fused_node_symbolic(op, As, j, I) for j in op.children[n]]
    return _fused_operations[op.operation_index[n]](args...)
end
function _fused_scalarize(
        op::FusedStencil{N, S}, x::SymbolicUtils.BasicSymbolic{T},
        ::Val{toplevel}
    ) where {N, S, T, toplevel}
    As = map(a -> SymbolicUtils.scalarize(a, Val{toplevel}()), arguments(x))
    out = Array{SymbolicUtils.BasicSymbolic{T}, N}(undef, S)
    for I in CartesianIndices(S)
        v = _fused_node_symbolic(op, As, length(op.kind), I)
        out[I] = v isa SymbolicUtils.BasicSymbolic ? v : SymbolicUtils.Const{T}(v)
    end
    return out
end
function Base.show(io::IO, op::FusedStencil{N, S}) where {N, S}
    return print(
        io, "FusedStencil(", S, ", ", count(==(:shifted), op.kind), " taps, ",
        count(==(:operation), op.kind), " ops)"
    )
end

_is_broadcast(ex) = iscall(ex) && (operation(ex) === broadcast || operation(ex) === mol_broadcast)

function _array_size(ex)
    sh = SymbolicUtils.shape(ex)
    (sh isa SymbolicUtils.Unknown || !SymbolicUtils.is_array_shape(sh)) && return nothing
    return Tuple(map(length, sh))
end

function _contains_derivative(ex)
    ex = Symbolics.unwrap(ex)
    ex isa SymbolicUtils.BasicSymbolic || return false
    if SymbolicUtils.isarrayop(ex)
        return ex.term !== nothing && _contains_derivative(ex.term)
    end
    iscall(ex) || return false
    operation(ex) isa Differential && return true
    return any(_contains_derivative, arguments(ex))
end

function _slice_parts(sl, N)
    sl = Symbolics.unwrap(sl)
    sl isa SymbolicUtils.BasicSymbolic || return nothing
    if SymbolicUtils.isarrayop(sl)
        tm = sl.term
        tm === nothing && return nothing
        return _slice_parts(tm, N)
    end
    (iscall(sl) && operation(sl) === getindex) || return nothing
    args = arguments(sl)
    length(args) == N + 1 || return nothing
    rs = map(unwrap_const, args[2:end])
    all(r -> r isa AbstractUnitRange{<:Integer}, rs) || return nothing
    base = Symbolics.unwrap(args[1])
    nested = SymbolicUtils.isarrayop(base) || (iscall(base) && operation(base) === getindex)
    if !nested
        T = SymbolicUtils.symtype(base)
        (T <: AbstractArray && ndims(T) == N) || return nothing
        return (base, NTuple{N, UnitRange{Int}}(map(r -> UnitRange{Int}(r), rs)))
    end
    inner = _slice_parts(base, N)
    inner === nothing && return nothing
    arr, r0 = inner
    all(d -> last(rs[d]) <= length(r0[d]), 1:N) || return nothing
    return (arr, NTuple{N, UnitRange{Int}}(ntuple(d -> UnitRange{Int}(r0[d][rs[d]]), N)))
end

mutable struct _FusedStencilBuilder{N}
    S::NTuple{N, Int}
    kind::Vector{Symbol}
    operation_index::Vector{Int}
    children::Vector{Vector{Int}}
    source::Vector{Int}
    ranges::Vector{NTuple{N, UnitRange{Int}}}
    symbolic_value::Vector{Any}
    constant_arrays::Vector{Any}
    args::Vector{Any}
end
_FusedStencilBuilder(S::NTuple{N, Int}) where {N} = _FusedStencilBuilder{N}(
    S, Symbol[], Int[], Vector{Int}[], Int[], NTuple{N, UnitRange{Int}}[], Any[], Any[], Any[]
)

function _push_fused_node!(
        b::_FusedStencilBuilder{N}, kind; operation_index = 0, children = Int[], source = 0,
        ranges = ntuple(_ -> 1:0, N), symbolic_value = nothing
    ) where {N}
    push!(b.kind, kind)
    push!(b.operation_index, operation_index)
    push!(b.children, children)
    push!(b.source, source)
    push!(b.ranges, ranges)
    push!(b.symbolic_value, symbolic_value)
    return length(b.kind)
end
function _fused_argument_index!(b, a)
    p = findfirst(x -> isequal(x, a), b.args)
    p === nothing && (push!(b.args, a); p = length(b.args))
    return p
end

function _fused_node!(b::_FusedStencilBuilder{N}, ex) where {N}
    ex = Symbolics.unwrap(ex)
    if !(ex isa SymbolicUtils.BasicSymbolic)
        ex isa Real || throw(ArgumentError("FusedStencil: constant $ex is not real"))
        return _push_fused_node!(b, :constant; symbolic_value = ex)
    end
    if SymbolicUtils.isconst(ex)
        v = unwrap_const(ex)
        v isa Real && return _push_fused_node!(b, :constant; symbolic_value = v)
        if v isa AbstractArray{<:Real} && size(v) == b.S
            push!(b.constant_arrays, Array(v))
            return _push_fused_node!(b, :constant_array; source = length(b.constant_arrays))
        end
        throw(ArgumentError("FusedStencil: unsupported constant"))
    end
    sh = _array_size(ex)
    if sh === nothing
        return _push_fused_node!(b, :scalar; source = _fused_argument_index!(b, ex))
    end
    sh == b.S || throw(ArgumentError("FusedStencil: shape $sh, expected $(b.S)"))
    parts = _slice_parts(ex, N)
    if parts !== nothing
        arr, rs = parts
        return _push_fused_node!(b, :shifted; source = _fused_argument_index!(b, arr), ranges = rs)
    end
    if _is_broadcast(ex)
        args = arguments(ex)
        f = unwrap_const(args[1])
        fc = findfirst(g -> g === f, _fused_operations)
        if fc !== nothing && all(a -> (s = _array_size(Symbolics.unwrap(a)); s === nothing || s == b.S), args[2:end])
            children = [_fused_node!(b, a) for a in args[2:end]]
            return _push_fused_node!(b, :operation; operation_index = fc, children)
        end
    end
    return _push_fused_node!(b, :array; source = _fused_argument_index!(b, fuse_stencils(ex)))
end

function fuse_stencils(ex)
    ex = Symbolics.unwrap(ex)
    ex isa SymbolicUtils.BasicSymbolic || return ex
    (SymbolicUtils.isarrayop(ex) || iscall(ex)) || return ex
    _contains_derivative(ex) && return _fuse_arguments(ex)
    S = _array_size(ex)
    if S !== nothing && _is_broadcast(ex)
        f = unwrap_const(arguments(ex)[1])
        if any(g -> g === f, _fused_operations)
            b = _FusedStencilBuilder(S)
            ok = try
                _fused_node!(b, ex)
                true
            catch e
                e isa ArgumentError || rethrow(e)
                false
            end
            if ok && count(==(:operation), b.kind) >= 1 && all(>(0), S)
                N = length(S)
                kern = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
                    @__MODULE__, @__MODULE__, _fused_loop_expr(b, N, S, length(b.args))
                )
                C = Tuple(b.constant_arrays)
                op = FusedStencil{N, S, typeof(kern), typeof(C)}(
                    b.kind, b.operation_index, b.children, b.source, b.ranges, b.symbolic_value, C, kern
                )
                sh = SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, S))
                return SymbolicUtils.term(op, b.args...; type = Array{Real, N}, shape = sh)
            end
        end
    end
    return _fuse_arguments(ex)
end

function _fuse_arguments(ex)
    SymbolicUtils.isarrayop(ex) && return ex
    iscall(ex) || return ex
    operation(ex) isa Differential && return ex
    args = arguments(ex)
    newargs = map(fuse_stencils, args)
    all(map(===, args, newargs)) && return ex
    return SymbolicUtils.Term{SymbolicUtils.SymReal}(
        operation(ex), collect(Any, newargs); type = SymbolicUtils.symtype(ex),
        shape = SymbolicUtils.shape(ex), metadata = TermInterface.metadata(ex)
    )
end
fuse_stencils(eq::Equation) = Equation(fuse_stencils(eq.lhs), fuse_stencils(eq.rhs))

hermite_finalize_equation(eq) = array_noinline_broadcasts(_fuse_or_keep(eq))

function _fuse_or_keep(eq)
    try
        return fuse_stencils(eq)
    catch e
        e isa InterruptException && rethrow(e)
        @debug "Keeping the unfused equation $eq: $(sprint(showerror, e))"
        return eq
    end
end
