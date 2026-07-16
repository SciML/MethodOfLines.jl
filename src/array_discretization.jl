# Array (slice/broadcast) discretization strategy, see issue #428.
#
# Instead of one scalar equation per interior grid point, the interior of each PDE is
# emitted as a single symbolic array equation over slices of the underlying array
# variables, e.g. for the 1D heat equation with second order approximation:
#
#   D(u[2:n-1]) ~ (u[1:n-2] .- 2u[2:n-1] .+ u[3:n]) ./ dx^2
#
# Boundary, extrapolation and corner equations reuse the scalar machinery, as do any
# interior points close enough to a boundary that their stencil differs from the
# translation-invariant interior stencil (the "frame").
#
# When an equation contains a pattern that is not representable as slice broadcasts
# (WENO/functional advection schemes, nonlinear or spherical laplacians, integrals,
# mixed derivatives, interface/periodic BCs, boundary values appearing in the interior
# equation, callbacks, staggered grids, variables of differing dimensionality), the
# whole equation falls back to pointwise scalar discretization, producing numerics
# identical to `ScalarizedDiscretization`.

struct ArrayDiscretizationFallback <: Exception
    msg::String
end

function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
        discretization::MOLFiniteDifference{G, D}
    ) where {G, D <: ArrayDiscretization}
    # Boundary handling is identical to the scalarized strategy
    boundaryvalfuncs = generate_boundary_val_funcs(
        s, depvars, bcmap, indexmap, derivweights
    )
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    for boundary in eqvarbcs
        generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end
    generate_extrap_eqs!(disc_state, pde, eqvar, s, derivweights, interiormap, bcmap)
    generate_corner_eqs!(disc_state, s, interiormap, ndims(s.discvars[eqvar]), eqvar)

    interior = interiormap.I[pde]
    eqs = if length(interior) == 0
        II = CartesianIndex()
        [
            discretize_equation_at_point(
                II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap, boundaryvalfuncs
            ),
        ]
    else
        try
            discretize_equation_array_form(
                pde, interior, s, depvars, derivweights, bcmap,
                eqvar, indexmap, boundaryvalfuncs
            )
        catch e
            e isa ArrayDiscretizationFallback || rethrow(e)
            @debug "ArrayDiscretization falling back to pointwise discretization for $pde: $(e.msg)"
            vec(
                map(interior) do II
                    discretize_equation_at_point(
                        II, s, depvars, pde, derivweights, bcmap,
                        eqvar, indexmap, boundaryvalfuncs
                    )
                end
            )
        end
    end

    return vcat!(disc_state.eqs, eqs)
end

"""
    discretize_equation_array_form(pde, interior, s, depvars, derivweights, bcmap,
                                   eqvar, indexmap, boundaryvalfuncs)

Discretize the interior of `pde` as a single symbolic array equation over slices of the
discretized dependent variables, plus pointwise scalar equations for any interior points
whose stencils differ from the translation-invariant interior stencil. Throws
`ArrayDiscretizationFallback` when `pde` contains a pattern that cannot be represented
this way.
"""
function discretize_equation_array_form(
        pde, interior, s, depvars, derivweights, bcmap,
        eqvar, indexmap, boundaryvalfuncs
    )
    get_grid_type(s) <: StaggeredGrid &&
        throw(ArrayDiscretizationFallback("staggered grids are not supported"))
    derivweights.advection_scheme isa UpwindScheme ||
        throw(ArrayDiscretizationFallback("only UpwindScheme advection is supported"))

    args = ivs(eqvar, s)
    for u in depvars
        isequal(ivs(u, s), args) ||
            throw(ArrayDiscretizationFallback("variables of differing dimensionality"))
        for x in args
            isempty(filter_interfaces(bcmap[operation(u)][x])) ||
                throw(ArrayDiscretizationFallback("interface/periodic boundary conditions"))
        end
    end

    pdeorders = Dict(x => d_orders(x, [pde]) for x in args)
    core = array_core_region(interior, s, args, pdeorders, derivweights, indexmap)
    length(core) == 0 && throw(ArrayDiscretizationFallback("empty core region"))

    # Probe the special-case rulesets at a representative core point. Several of these
    # generators return candidate rules unconditionally; the scalar path only applies a
    # special scheme when a rule key occurs in the equation, so fall back exactly when
    # one does. Any firing rule means a scheme with no slice representation here yet.
    II0 = first(core)
    terms = split_terms(pde, s.x̄)
    special_rules = vcat(
        vec(generate_mixed_rules(II0, s, depvars, derivweights, bcmap, indexmap, terms)),
        vec(generate_nonlinlap_rules(II0, s, depvars, derivweights, bcmap, indexmap, terms)),
        vec(
            generate_spherical_diffusion_rules(
                II0, s, depvars, derivweights, bcmap,
                indexmap, split_additive_terms(pde)
            )
        ),
        vec(generate_euler_integration_rules(II0, s, depvars, indexmap, terms)),
        vec(generate_whole_domain_integration_rules(II0, s, depvars, indexmap, terms)),
        vec(generate_cb_rules(II0, s, depvars, derivweights, bcmap, indexmap, terms)),
        mapreduce(f -> f(II0), vcat, boundaryvalfuncs, init = [])
    )
    for r in special_rules
        (subsmatch(pde.lhs, r) || subsmatch(pde.rhs, r)) &&
            throw(ArrayDiscretizationFallback("unsupported pattern $(r.first)"))
    end

    ranges = Dict(indexmap[x] => first(core)[indexmap[x]]:last(core)[indexmap[x]] for x in args)

    # Ordered substitution rules; the first matching rule wins at each node.
    varrules = [safe_unwrap(u) => array_slice(u, s, ranges, indexmap) for u in depvars]
    gridrules = [
        safe_unwrap(x) => array_grid_vals(x, s, ranges, indexmap, length(args))
            for x in args
    ]
    derivrules = array_cartesian_rules(s, depvars, pdeorders, derivweights, ranges, indexmap)
    windrules = array_winding_rules(
        terms, s, depvars, pdeorders, derivweights, ranges, indexmap,
        vcat(varrules, gridrules)
    )
    ctx = ArrayifyContext(vcat(windrules, derivrules, varrules, gridrules), s.time)

    lhs = arrayify(pde.lhs, ctx)
    rhs = arrayify(pde.rhs, ctx)
    # `~` cannot equate an array with a scalar; the system is cardinalized so the rhs is
    # (a scalar) 0 and the lhs holds the whole residual.
    if is_array_valued(lhs) && !is_array_valued(rhs)
        isequal(safe_unwrap(rhs), 0) ||
            throw(ArrayDiscretizationFallback("array lhs with non-zero scalar rhs"))
        rhs = zeros(size(core))
    elseif !is_array_valued(lhs) && is_array_valued(rhs)
        isequal(safe_unwrap(lhs), 0) ||
            throw(ArrayDiscretizationFallback("array rhs with non-zero scalar lhs"))
        lhs = zeros(size(core))
    elseif !is_array_valued(lhs) && !is_array_valued(rhs)
        throw(ArrayDiscretizationFallback("equation contains no discretizable terms"))
    end
    core_eq = lhs ~ rhs

    frame = setdiff(vec(collect(interior)), vec(collect(core)))
    frame_eqs = map(frame) do II
        discretize_equation_at_point(
            II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap, boundaryvalfuncs
        )
    end
    return vcat([core_eq], frame_eqs)
end

"""
Compute the subbox of the interior on which every derivative appearing in the equation
resolves to the translation-invariant interior stencil, mirroring the branch conditions
in `central_difference_weights_and_stencil` and `_upwind_difference`.
"""
function array_core_region(interior, s, args, pdeorders, derivweights, indexmap)
    lo = collect(Tuple(first(interior)))
    hi = collect(Tuple(last(interior)))
    for x in args
        j = indexmap[x]
        n = length(s, x)
        for d in pdeorders[x]
            if iseven(d)
                Dop = derivweights.map[Differential(x)^d]
                lo[j] = max(lo[j], Dop.boundary_point_count + 1)
                hi[j] = min(hi[j], n - Dop.boundary_point_count)
            else
                Dneg = derivweights.windmap[1][Differential(x)^d]
                Dpos = derivweights.windmap[2][Differential(x)^d]
                # Positive winding taps (-stencil_length+1):0, boundary branch at
                # II <= offside; negative winding taps 0:(stencil_length-1), boundary
                # branch at II > n - boundary_point_count.
                lo[j] = max(lo[j], Dpos.offside + 1, Dpos.stencil_length)
                hi[j] = min(hi[j], n - Dneg.boundary_point_count, n - Dneg.stencil_length + 1)
            end
        end
    end
    any(map((l, h) -> l > h, lo, hi)) && return CartesianIndices(())[1:0]
    return CartesianIndices(Tuple(map((l, h) -> l:h, lo, hi)))
end

"""
The underlying (unscalarized) array variable of which `s.discvars[u]` holds the elements.
"""
function array_variable(u, s)
    el = safe_unwrap(first(vec(s.discvars[u])))
    (iscall(el) && operation(el) === getindex) ||
        throw(ArrayDiscretizationFallback("discrete variable for $u is not an array variable"))
    return Symbolics.wrap(first(arguments(el)))
end

"""
A slice of the array variable for `u` over the core region, optionally shifted by
`offset` in the dimension of `shiftx`.
"""
function array_slice(u, s, ranges, indexmap; shiftx = nothing, offset = 0)
    arr = array_variable(depvar(u, s), s)
    rs = map(ivs(depvar(u, s), s)) do y
        r = ranges[indexmap[y]]
        (shiftx !== nothing && isequal(y, shiftx)) ? (r .+ offset) : r
    end
    return arr[rs...]
end

"""
The numeric grid values of `x` over the core region, shaped to broadcast along the
dimension of `x` in an `N`-dimensional array expression.
"""
function array_grid_vals(x, s, ranges, indexmap, N)
    j = indexmap[x]
    vals = collect(s.grid[x][ranges[j]])
    N == 1 && return vals
    return reshape(vals, ntuple(i -> i == j ? length(vals) : 1, N))
end

"""
Per-point stencil weights as a broadcastable numeric array along dimension `j` of `N`,
for nonuniform grids where the interior weights vary from point to point.
`getweights(i)` returns the weight `SVector` at grid index `i`.
"""
function array_weight_vals(getweights, k, rng, j, N)
    vals = [getweights(i)[k] for i in rng]
    N == 1 && return vals
    return reshape(vals, ntuple(i -> i == j ? length(vals) : 1, N))
end

"""
Broadcasted weighted sum of shifted slices: the array-form analogue of `sym_dot`.
"""
function array_stencil(weights, slices)
    wterms = [Broadcast.materialize(Broadcast.broadcasted(*, w, sl)) for (w, sl) in zip(weights, slices)]
    length(wterms) == 1 && return wterms[1]
    return Broadcast.materialize(Broadcast.broadcasted(+, wterms...))
end

"""
Array form of `central_difference` on the core region for the even order derivative
`(Differential(x)^d)(u)`.
"""
function array_central_difference(Dop, s, u, x, d, ranges, indexmap)
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]
    taps = half_range(Dop.stencil_length)
    slices = [array_slice(u, s, ranges, indexmap; shiftx = x, offset = k) for k in taps]
    weights = if Dop.dx isa Number
        collect(Dop.stencil_coefs)
    else
        bpc = Dop.boundary_point_count
        [
            array_weight_vals(i -> Dop.stencil_coefs[i - bpc], k, rng, j, N)
                for k in eachindex(taps)
        ]
    end
    return array_stencil(weights, slices)
end

@inline function array_cartesian_rules(
        s, depvars, pdeorders, derivweights, ranges, indexmap
    )
    rules = Pair[]
    for u in depvars, x in ivs(depvar(u, s), s)
        for d in filter(iseven, pdeorders[x])
            Dop = derivweights.map[Differential(x)^d]
            push!(
                rules,
                safe_unwrap((Differential(x)^d)(u)) => array_central_difference(
                    Dop, s, u, x, d, ranges, indexmap
                )
            )
        end
    end
    return rules
end

"""
Array form of `upwind_difference` on the core region, for one winding direction.
"""
function array_upwind_difference(s, u, x, d, derivweights, ranges, indexmap, ispositive)
    Dop = ispositive ? derivweights.windmap[2][Differential(x)^d] :
        derivweights.windmap[1][Differential(x)^d]
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]
    taps = ispositive ? ((-Dop.stencil_length + 1):0) : (0:(Dop.stencil_length - 1))
    slices = [array_slice(u, s, ranges, indexmap; shiftx = x, offset = k) for k in taps]
    weights = if Dop.dx isa Number
        collect(Dop.stencil_coefs)
    else
        # Mirrors `_upwind_difference` for nonuniform grids, where the interior weights
        # are indexed as stencil_coefs[II[j]] (negative) / stencil_coefs[II[j] - offside]
        # (positive, with offside == 0 for nonuniform operators).
        [
            array_weight_vals(i -> Dop.stencil_coefs[i - Dop.offside], k, rng, j, N)
                for k in eachindex(taps)
        ]
    end
    return array_stencil(weights, slices)
end

"""
Array form of the winding selection for an odd derivative multiplied by expression
`expr`. The scalar path emits `ifelse(coef > 0, coef*pos, coef*neg)` pointwise; `ifelse`
cannot currently be broadcast over symbolic array conditions, so this uses the
numerically equivalent `max(coef, 0)*pos + min(coef, 0)*neg`.
"""
function array_winding_select(expr, s, u, x, d, derivweights, ranges, indexmap, coefctx)
    coef = arrayify(expr, coefctx)
    pos = array_upwind_difference(s, u, x, d, derivweights, ranges, indexmap, true)
    neg = array_upwind_difference(s, u, x, d, derivweights, ranges, indexmap, false)
    bcast(op, args...) = Broadcast.materialize(Broadcast.broadcasted(op, args...))
    return bcast(
        +,
        bcast(*, bcast(max, coef, 0), pos),
        bcast(*, bcast(min, coef, 0), neg)
    )
end

@inline function array_winding_rules(
        terms, s, depvars, pdeorders, derivweights, ranges, indexmap, baserules
    )
    coefctx = ArrayifyContext(baserules, s.time)
    ruleobjs = []
    for u in depvars, x in ivs(depvar(u, s), s)
        for d in filter(isodd, pdeorders[x])
            push!(
                ruleobjs,
                @rule *(
                    ~~a, $(Differential(x)^d)(u), ~~b
                ) => array_winding_select(
                    *(~a..., ~b...), s, u, x, d, derivweights, ranges, indexmap, coefctx
                )
            )
            push!(
                ruleobjs,
                @rule /(
                    *(~~a, $(Differential(x)^d)(u), ~~b), ~c
                ) => array_winding_select(
                    *(~a..., ~b...) / ~c, s, u, x, d, derivweights,
                    ranges, indexmap, coefctx
                )
            )
        end
    end

    windrules = Pair[]
    for t in terms
        for r in ruleobjs
            v = r(t)
            if v !== nothing
                push!(windrules, safe_unwrap(t) => v)
            end
        end
    end

    # Default rules for bare odd derivatives (no coefficient): positive winding,
    # mirroring the tail of `generate_winding_rules`.
    for u in depvars, x in ivs(depvar(u, s), s)
        for d in filter(isodd, pdeorders[x])
            push!(
                windrules,
                safe_unwrap((Differential(x)^d)(u)) => array_upwind_difference(
                    s, u, x, d, derivweights, ranges, indexmap, true
                )
            )
        end
    end
    return windrules
end

struct ArrayifyContext
    rules::Vector{<:Pair}
    time::Any
end

function is_array_valued(x)
    x isa AbstractArray && return true
    u = safe_unwrap(x)
    u isa SymbolicUtils.Symbolic || return false
    return SymbolicUtils.symtype(u) <: AbstractArray
end

"""
    arrayify(expr, ctx)

Broadcast-aware substitution: rebuild `expr` bottom-up, replacing any subterm that
matches a rule in `ctx.rules` (first match wins) and broadcasting any operation that
receives an array-valued argument. Time differentials are applied directly to their
(array-valued) arguments; any spatial differential that survives the rules means the
expression contains a scheme this path does not support, so fall back.
"""
function arrayify(expr, ctx)
    expr = safe_unwrap(expr)
    for (k, v) in ctx.rules
        isequal(expr, k) && return v
    end
    iscall(expr) || return Symbolics.wrap(expr)
    op = operation(expr)
    if op isa Differential
        isequal(op.x, ctx.time) ||
            throw(ArrayDiscretizationFallback("unhandled spatial derivative in $expr"))
        arg = arrayify(only(arguments(expr)), ctx)
        return op(arg)
    end
    newargs = [arrayify(a, ctx) for a in arguments(expr)]
    if any(is_array_valued, newargs)
        return Broadcast.materialize(Broadcast.broadcasted(op, newargs...))
    else
        return op(newargs...)
    end
end
