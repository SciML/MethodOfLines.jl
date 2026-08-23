"""
The substitution key for the two-direction mixed derivative
`(Differential(x)^m * Differential(y)^n)(u)`.

This is the form both `generate_mixed_rules` and the array path match. The historical
`m = n = 1` spelling `(Differential(x) * Differential(y))(u)` is `isequal` to
`Differential(x)^1 * Differential(y)^1` in current Symbolics; `mixed_derivative_keys`
still emits both spellings so a mismatch cannot drop a `(1, 1)` term.
"""
mixed_derivative_key(u, x, m, y, n) = (Differential(x)^m * Differential(y)^n)(u)

"""
Substitution keys for `(Differential(x)^m * Differential(y)^n)(u)`, including the
unpowered `(Differential(x) * Differential(y))(u)` spelling when `m = n = 1`.
"""
function mixed_derivative_keys(u, x, m, y, n)
    keys = Any[safe_unwrap(mixed_derivative_key(u, x, m, y, n))]
    if m == 1 && n == 1
        k0 = safe_unwrap((Differential(x) * Differential(y))(u))
        any(k -> isequal(k, k0), keys) || push!(keys, k0)
    end
    return keys
end

"""
Map each spatial direction to the centered mixed-derivative orders that reach along it.
"""
function mixed_orders_by_direction(mixedterms)
    orders = Dict{Any, Set{Int}}()
    for (_, x, mx, y, my) in mixedterms
        push!(get!(Set{Int}, orders, x), mx)
        push!(get!(Set{Int}, orders, y), my)
    end
    return orders
end

"""
Performs a mixed centered difference in `x` centered at index `II` of `u`
ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes
"""
function mixed_central_difference((Dx, Dy), II, s, (xbs, ybs), (jx, ky), u, ufunc)
    j, x = jx
    k, y = ky
    xweights, xItap = central_difference_weights_and_stencil(Dx, II, s, xbs, jx, u)
    yweights, yItap = central_difference_weights_and_stencil(Dy, II, s, ybs, ky, u)
    # TODO: Fix interface bcs

    out = sum(zip(xweights, xItap)) do (wx, xI)
        sum(zip(yweights, yItap)) do (wy, yI)
            xoffset = xI - II
            yoffset = yI - II
            I = II + xoffset + yoffset
            wx * wy * ufunc(u, I, x)
        end
    end

    return out
end

@inline function generate_mixed_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms
    )
    central_ufunc(u, I, x) = s.discvars[u][I]
    function emit(u, x, mx, y, my, key)
        return key => mixed_central_difference(
            (
                derivweights.map[Differential(x)^mx],
                derivweights.map[Differential(y)^my],
            ),
            Idx(II, s, u, indexmap),
            s,
            (
                filter_interfaces(bcmap[operation(u)][x]),
                filter_interfaces(bcmap[operation(u)][y]),
            ),
            ((x2i(s, u, x), x), (x2i(s, u, y), y)),
            u,
            central_ufunc
        )
    end
    rules = []
    for u in depvars
        xs = ivs(u, s)
        for x in xs, y in remove(xs, unwrap(x))
            # Always emit the historical `(1, 1)` key so existing first-order mixed
            # substitutions stay bit-identical, including when `orders` omits 1.
            push!(
                rules,
                emit(u, x, 1, y, 1, (Differential(x) * Differential(y))(u))
            )
            for mx in get(derivweights.orders, x, Int[]),
                    my in get(derivweights.orders, y, Int[])
                (mx == 1 && my == 1) && continue
                push!(
                    rules,
                    emit(u, x, mx, y, my, mixed_derivative_key(u, x, mx, y, my))
                )
            end
        end
    end
    return rules
end
