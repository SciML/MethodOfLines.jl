"""
    _hermite_operation(a)

The operation of an applied variable `u(t, x)`, or `a` itself for a bare `u`.
"""
@inline function _hermite_operation(a)
    av = unwrap(a)
    return iscall(av) ? operation(av) : av
end

"""
    _same_hermite_variable(a, b)

Whether `a` and `b` are the same dependent variable, applied (`u(t, x)`) or bare (`u`).
"""
@inline function _same_hermite_variable(a, b)::Bool
    return isequal(_hermite_operation(a), _hermite_operation(b))
end

"""
    _resolve_hermite_variable(v, s::DiscreteSpace)

Return the key in `s.discvars` that matches `v`, or `nothing`.
"""
@inline function _resolve_hermite_variable(v, s)
    haskey(s.discvars, v) && return v
    for candidate in keys(s.discvars)
        _same_hermite_variable(candidate, v) && return candidate
    end
    return nothing
end

"""
    _hermite_is_diffusive(scheme::HermiteScheme, u, x)

Whether the PDE of `u` is diffusive in `x` (recorded by the transformation in
`scheme.diffusive`), so its stencils near `x` boundaries use the wider windows of
[`_hermite_boundary_stencil`](@ref).
"""
_hermite_is_diffusive(scheme::HermiteScheme, u, x) =
    any(k -> _same_hermite_variable(k[1], u) && isequal(k[2], x), scheme.diffusive)

"""
    _hermite_partner(scheme::HermiteScheme, u, x, s::DiscreteSpace)

Look up the derivative-partner variable for `u` in direction `x`.
Returns the resolved key into `s.discvars`, or `nothing` if `u` is not mapped.
"""
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

"""
    _hermite_weights_and_taps(D, H, II, s, bs, jx, u; wide = false)

Compute the HFD weight vectors `(d_weights, e_weights)` and stencil tap indices
for the derivative operator `D` at grid index `II`.  Reuses
`central_difference_weights_and_stencil` for tap selection; interior uniform
points use the precomputed weights in `H`, boundary/nonuniform points recompute
via `calculate_weights` on the taps of [`_hermite_boundary_stencil`](@ref).
"""
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

Grid points of the Hermite stencil for `D` at a point within `boundary_point_count` of a
boundary of `x`, in place of MOL's one-sided points `Itap`.

A Hermite stencil uses the values and the first derivatives at its points, so on `m` points
it has order `2m - d` for a derivative of order `d`. On MOL's `d + approx_order` one-sided
points this order and the weights are large, and the semi-discrete system is unstable
above `approx_order = 2`. Instead, the point `k` steps from the boundary uses the window of
`2k + 1` points centred on it, clipped to the grid, with at least `m = d + 1` and at most
`length(Itap)` points, so the stencil widens gradually into the interior one. The windows
must be centred: the `d + 1` points nearest the boundary are unstable at high
`approx_order`. MOL's points are kept at `approx_order = 2`, except for `d ≥ 3` (the
`Dxxx(u)` in the partner equation of a wave equation), where they are unstable too.

With `d + 1` points, the two points nearest the boundary have order `d + 2`, which limits
the global order. `wide = true`, for a variable whose PDE is diffusive in `x` (see
[`_hermite_is_diffusive`](@ref)), raises the minimum to `m = cld(approx_order + d, 2) + 1`
points, of boundary order `2m - d ≥ approx_order + 2`. It is not used when the PDE has a
first derivative in `x`, where these wider windows are unstable.
"""
function _hermite_boundary_stencil(D, II, s, j, x, Itap; wide = false)
    d = D.derivative_order
    m = wide ? cld(D.approximation_order + d, 2) + 1 : d + 1
    n = length(s, x)
    i = II[j]
    bpc = D.boundary_point_count
    lower = i <= bpc
    ((lower || i > n - bpc) && (length(Itap) > d + 2 || d >= 3) && length(Itap) > m) ||
        return Itap
    k = lower ? i - 1 : n - i
    w = min(max(m, 2k + 1), length(Itap))
    first = clamp(lower ? i - cld(w - 1, 2) : i - fld(w - 1, 2), 1, n - w + 1)
    return [CartesianIndex(Base.setindex(Tuple(Itap[1]), l, j)) for l in first:(first + w - 1)]
end

"""
    hermite_central_difference(D, H, II, s, bs, jx, u, v; wide = false)

Return the symbolic HFD approximation `sym_dot(D_w, u_taps) + sym_dot(E_w, v_taps)`
for the derivative described by operator `D` at index `II`.
"""
@inline function hermite_central_difference(D, H, II, s, bs, jx, u, v; wide = false)
    dweights, eweights, Itap = _hermite_weights_and_taps(D, H, II, s, bs, jx, u; wide)
    u_vals = [s.discvars[u][I] for I in Itap]
    v_vals = [s.discvars[v][I] for I in Itap]
    return sym_dot(dweights, u_vals) + sym_dot(eweights, v_vals)
end

"""
    hermite_partner_derivative_rule(derivweights, s, v_, II, x_)

The rule `Dx(v) => w_xx` for a boundary condition on the partner `v = Dx(w)` at grid index
`II`, with the Hermite stencil of `w` and `v` as in the PDE rows; `nothing` if `v_` is not a
partner in `x_`.

A Dirichlet condition on `w` gives its partner the derived condition `Dx(v) ~ w_t` (the PDE
at the boundary). MOL's one-sided first difference of `v` alone has order `approx_order`,
below that of the PDE rows; the Hermite stencil uses `w` and `v` together and has their order.
"""
function hermite_partner_derivative_rule(derivweights, s, v_, II, x_)
    v = depvar(v_, s)
    p = _hermite_partner_second_derivative(derivweights, s, v, x_)
    p === nothing && return nothing
    rhs = hermite_central_difference(
        p.D, p.H, II, s, [], (x2i(s, p.w, x_), x_), p.w, v; wide = p.wide
    )
    return (Differential(x_)^1)(v_) => rhs
end

"""
    array_hermite_partner_derivative(derivweights, s, v, x_, ranges, indexmap, II0)

Array (face slice) form of [`hermite_partner_derivative_rule`](@ref): the Hermite
approximation of `w_xx` for `Dx(v)` on a whole boundary face of `x_`, where `v = Dx(w)`.
The weights and taps are those at the representative point `II0`; they depend only on the
index along `x_`, which is constant on the face. `nothing` if `v` is not a partner in `x_`.
"""
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

"""
    _hermite_partner_second_derivative(derivweights, s, v, x)

For a derivative partner `v = Dx(w)`, the `w`, the operator `D` and Hermite weights `H` of
`Dx^2` and whether `w`'s PDE is diffusive in `x`, as `(; w, D, H, wide)`; `nothing` if the
scheme is not a `HermiteScheme`, `v` is not a partner in `x`, or `Dx^2` has no weights.
"""
function _hermite_partner_second_derivative(derivweights, s, v, x)
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

"""
    _hermite_primary(scheme::HermiteScheme, v, x, s)

The variable `w` whose partner in direction `x` is `v` (`v = Dx(w)`), resolved as a key of
`s.discvars`, or `nothing`.
"""
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

Generate replacement rules for all spatial derivatives at index `II`.

For each dependent variable and spatial direction:
  - If a partner is mapped via `HermiteScheme`, use HFD weights for all orders.
  - Otherwise, fall back to standard central differences.

This replaces `generate_cartesian_rules` when `advection_scheme isa HermiteScheme`.
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

Rules for the mixed derivative `u_xy` of every `u` with derivative partners `v = u_x` and
`w = u_y`, in each form the transformed system can contain: `Dx(Dy(u))`, `Dy(Dx(u))`,
`Dy(v)` and `Dx(w)`. All get

    u_xy ≈ δy v + δx w - δx δy u,

where `δx`, `δy` are first-derivative stencils: MOL's in the interior, and near boundaries
those of [`_mixed_first_difference`](@ref). Writing `δx = (1 + Ex) ∂x`, the combination
equals `(1 - Ex Ey) u_xy`, so its error is the product of the two stencil errors: order
`2 approx_order` in the interior, against `approx_order` for `δx δy u` or `δy v` alone. These rules come after the
pure-derivative rules, so they replace the ordinary `Dy(v)` difference.
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

"""
    _mixed_first_difference(D, II, s, bs, jx, u; tbcs = [])

Weights and grid points of the first difference `δx` in the `u_xy` combination of
[`hermite_mixed_derivative_rules`](@ref).

In the interior and on periodic axes this is MOL's stencil. Near a boundary with data
(`tbcs`, the non-periodic conditions of `u` in `x`, has one on that side) and for
`approx_order > 2`, it uses the centred window of [`_hermite_boundary_stencil`](@ref) with
ordinary (Fornberg) weights. The combination puts `δx w` into the equation of `w` itself,
and MOL's one-sided stencil, biased downwind next to an inflow boundary, gives a growing
mode next to an inflow corner. The lower order of the window costs little, since the error
of the combination is the product of the two stencil errors.

At a side without a condition (a free outflow), MOL's one-sided stencil is kept: there it
looks upstream, while the window's two-point difference at the last point is of first order
and limits the global order.
"""
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

"""
    _hermite_face_mixed_derivative(II, s, u, v, w, x, y, bcmap, indexmap)

`u_xy` at a point `II` on a boundary face of `x` where the normal partner `v = u_x` has a
condition `v ~ F` (Neumann/Robin data, see `hermite_transform!`) and the tangential partner
`w = u_y` has none, so `w` is evolved by its own equation there. Along the face `v = F`
holds exactly, hence `u_xy = Dy(F)`, which is discretized directly: `Dy(u)` becomes `w`,
`u` its grid value and `y` its grid coordinate. Neumann data give `Dy(g)`, a Robin
condition `u - u_x = r` gives `w - Dy(r)`; both exact.

The interior combination `δy v + δx w - δx δy u` uses a one-sided `δx w` here. At an
inflow face that adds a downwind difference of `w` to `w`'s own equation, which lowers the
order of 2D advection with Neumann or Robin inflow.

Returns `nothing` when `II` is not on such a face, or when `Dy(F)` contains derivatives or
variables other than `u` and `Dy(u)` (for example another dependent variable); the caller
then keeps the interior combination.
"""
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

"""
    _is_face_data(ex, s, y)

Whether `ex`, after the substitutions of [`_hermite_face_mixed_derivative`](@ref), is data
on the face: no derivative, no dependent-variable call other than grid values, and no `y`.
"""
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

"""
    array_validate_hermite(s, args, periodic)

Throw `ArrayFormFallback` where the Hermite array form does not apply: a non-uniform grid
(the pointwise path recomputes the weights at every point) or a two-domain interface (the
pointwise path rejects cross-variable interface stencils with an error of its own).
"""
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

"""
    _hermite_mixed_derivative_terms(s, depvars, derivweights)

The `(; u, v, w, x, y, keys)` for which the `u_xy` combination of
[`hermite_mixed_derivative_rules`](@ref) applies: `u` with partners `v = u_x` and `w = u_y`,
and the rule keys of that combination.
"""
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

"""
    _hermite_pde_has(pde, key)

Whether `key` occurs in either side of `pde`.
"""
_hermite_pde_has(pde, key) = subsmatch(pde.lhs, key => nothing) || subsmatch(pde.rhs, key => nothing)

"""
    array_hermite_mixed_orders!(mixedorders, pde, s, depvars, derivweights)

Register derivative order 1 in `x` and `y` for every variable the `u_xy` combination of
`pde` reads, so the points where its stencils are not the interior ones stay pointwise.

The `u_xy` combination reads first-difference stencils of `u`, `v` and `w` in both
directions, also where the equation itself has no derivative in one of them (the `uˍy`
equation of 2D advection contains `Dy(uˍx)` but nothing in `x`).
"""
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

Core-region rules of the Hermite array form, as `(mixedrules, derivrules)`.

In the core every derivative takes its interior stencil, so the pointwise rules of
[`generate_hermite_rules`](@ref) reduce to fixed weights applied to shifted slices: the
Hermite stencil `Σ D_k u[i+k] + Σ E_k v[i+k]` for a variable `u` with partner `v`, MOL's
centred stencil for a variable without one, and the `u_xy` combination from MOL's
first-difference stencils. The points outside the core stay pointwise, so the equations
are those of the pointwise path, spelled over slices. The mixed rules must take precedence
over MOL's tensor stencil for `Dx(Dy(u))`, as in the pointwise path. Throws
`ArrayFormFallback` if an interior Hermite weight set is missing.
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

"""
    array_hermite_corner_dirichlet(s, u, ranges, J, bcmap)

The corner point `ranges` (non-interior in the directions `J`) set to the Dirichlet data of
`u`, evaluated at the corner, as one slice equation; or `nothing`, so that the caller
extrapolates, unless every direction in `J` has a Dirichlet condition on `u` at this end and
none has an interface. The data of the meeting faces agree at the corner, so any one of
them gives the value.
"""
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
            sub = Dict(args[k] => s.grid[args[k]][first(ranges[k])] for k in 1:length(args))
            val = substitute(expand_derivatives(F), sub)
            _is_corner_data(Symbolics.unwrap(val), s, args) || return nothing
        end
    end
    val === nothing && return nothing
    arr = array_variable(u, s)
    return arr[ranges...] ~ reshape([val], ntuple(_ -> 1, length(args)))
end

"""
    _is_corner_data(ex, s, args)

Whether `ex` is data on a corner: no derivative, no grid value or call of a dependent
variable, and none of the spatial variables `args`.
"""
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
    FusedStencil{N, S, K}

One elementwise expression tree over a box of size `S`, evaluated in one pass instead of
as nested broadcasts over slices, where every slice is a copy and every broadcast a
temporary.

Nodes are stored children-first, each with a `kind`: `:constant`, `:shifted` (argument
`source` read at the shifted index `ranges`), `:array` (argument `source` read at the same
element), `:scalar`, `:constant_array`, or `:operation` (`_fused_operations[operation_index]`
applied to `children`). `kernel` is the tree compiled into a loop with one statement per
node ([`_fused_loop_expr`](@ref)); the node list is kept for scalarization. The loop applies
the same operations to the same operands in the same order as the nested broadcasts, with
n-ary `+` and `*` folded left, so the values are bit-for-bit the same.
"""
struct FusedStencil{N, S, K} <: Function
    kind::Vector{Symbol}
    operation_index::Vector{Int}
    children::Vector{Vector{Int}}
    source::Vector{Int}
    ranges::Vector{NTuple{N, UnitRange{Int}}}
    value::Vector{Float64}
    symbolic_value::Vector{Any}
    constant_arrays::Vector{Array{Float64, N}}
    kernel::K
end
Base.size(::FusedStencil{N, S}) where {N, S} = S

"""
    _fused_loop_expr(b, N, S, nargs)

The expression `function (out, C, A1, ..., An) ... end` that evaluates the node list of
builder `b` over the box `S`, with `i1` innermost. `C` is the vector of constant arrays,
passed as one argument so that the call does not splat a vector of unknown length.
"""
function _fused_loop_expr(b, N, S, nargs)
    A = [Symbol("A", k) for k in 1:nargs]
    is = [Symbol("i", d) for d in 1:N]
    v(n) = Symbol("v", n)
    stmts = Expr[]
    for n in eachindex(b.kind)
        k = b.kind[n]
        rhs = if k == :constant
            b.value[n]
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
        function (out, C, $(A...))
            @inbounds $loop
            return out
        end
    )
end

function (op::FusedStencil{N, S})(As::Vararg{Any, K}) where {N, S, K}
    T = promote_type(Float64, map(eltype, As)...)
    out = Array{T, N}(undef, S)
    return op.kernel(out, op.constant_arrays, As...)
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

"""
    _slice_parts(sl, N)

`(array, ranges)` of a plain slice `arr[r1, ..., rN]`, possibly nested or wrapped in an
`ArrayOp`, or `nothing`.
"""
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
    value::Vector{Float64}
    symbolic_value::Vector{Any}
    constant_arrays::Vector{Array{Float64, N}}
    args::Vector{Any}
end
_FusedStencilBuilder(S::NTuple{N, Int}) where {N} = _FusedStencilBuilder{N}(
    S, Symbol[], Int[], Vector{Int}[], Int[],
    NTuple{N, UnitRange{Int}}[], Float64[], Any[], Array{Float64, N}[], Any[]
)

function _push_fused_node!(
        b::_FusedStencilBuilder{N}, kind; operation_index = 0, children = Int[], source = 0,
        ranges = ntuple(_ -> 1:0, N), value = 0.0, symbolic_value = value
    ) where {N}
    push!(b.kind, kind)
    push!(b.operation_index, operation_index)
    push!(b.children, children)
    push!(b.source, source)
    push!(b.ranges, ranges)
    push!(b.value, Float64(value))
    push!(b.symbolic_value, symbolic_value)
    return length(b.kind)
end
function _fused_argument_index!(b, a)
    p = findfirst(x -> isequal(x, a), b.args)
    p === nothing && (push!(b.args, a); p = length(b.args))
    return p
end

"""
    _fused_node!(b::_FusedStencilBuilder, ex)

Push the node for `ex`, an array of size `b.S` or a scalar, and its subtree; return its
index. Array subexpressions that are not arithmetic broadcasts become argument nodes, with
their own subtrees fused first. Throws an `ArgumentError` for what cannot be fused.
"""
function _fused_node!(b::_FusedStencilBuilder{N}, ex) where {N}
    ex = Symbolics.unwrap(ex)
    if !(ex isa SymbolicUtils.BasicSymbolic)
        ex isa Real || throw(ArgumentError("FusedStencil: non-real constant $ex"))
        return _push_fused_node!(b, :constant; value = ex, symbolic_value = ex)
    end
    if SymbolicUtils.isconst(ex)
        v = unwrap_const(ex)
        v isa Real && return _push_fused_node!(b, :constant; value = v, symbolic_value = v)
        if v isa AbstractArray{<:Real} && size(v) == b.S
            push!(b.constant_arrays, Array{Float64, N}(v))
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

"""
    fuse_stencils(ex)

`ex` with each maximal arithmetic broadcast tree over arrays of one size replaced by one
[`FusedStencil`](@ref) call that evaluates the same tree in the same order. Subtrees with
time derivatives are left unchanged.
"""
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
            if ok && count(==(:operation), b.kind) >= 1
                N = length(S)
                kern = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
                    @__MODULE__, @__MODULE__, _fused_loop_expr(b, N, S, length(b.args))
                )
                op = FusedStencil{N, S, typeof(kern)}(
                    b.kind, b.operation_index, b.children, b.source, b.ranges, b.value, b.symbolic_value, b.constant_arrays, kern
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

"""
    hermite_finalize_equation(eq)

The last step for a HermiteScheme equation in `PDEBase.discretize_equation!`: fuse its slice
expressions ([`fuse_stencils`](@ref)), then spell the remaining broadcasts as `mol_broadcast`
like every other equation. If fusing fails, the equation is kept as it was.
"""
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
