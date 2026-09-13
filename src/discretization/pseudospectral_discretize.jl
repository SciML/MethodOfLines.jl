# `PseudospectralDiscretization` pipeline.
#
# The parsing, boundary-condition and system-construction machinery is shared
# with `MOLFiniteDifference`; the differences are confined to grid
# construction (`construct_discrete_space`), the derivative operators
# (`construct_differential_discretizer`), interior sizing (no ghost-point
# padding: spectral rows always span the whole direction, and higher-order
# periodic matching conditions are redundant), and equation generation. The
# interior is emitted in slice form (`discretize_spectral_array_form`), with each
# derivative a `SpectralApply` operator over the whole direction; boundary faces
# reuse `array_bc_eqs`. Patterns without a slice form fall back to the pointwise
# collocation rules in `schemes/pseudospectral/pseudospectral.jl`.

function PDEBase.interface_errors(
        pdesys::PDESystem, v::PDEBase.VariableMap, discretization::PseudospectralDiscretization
    )
    for x in v.x̄
        @assert haskey(discretization.dxs, Num(x))||haskey(discretization.dxs, x) "Variable $x has no collocation specification"
    end
    return
end

function PDEBase.check_boundarymap(
        boundarymap, v::PDEBase.VariableMap, discretization::PseudospectralDiscretization
    )
    bs = flatten_vardict(boundarymap)
    for b in bs
        if b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}
            isequal(b.x, b.x2) || throw(
                ArgumentError(
                    "PseudospectralDiscretization does not yet support multi-domain interface boundary conditions like $(b.eq), only same-variable periodic conditions are supported."
                )
            )
        end
    end
    for x in v.x̄
        spec = discretization.dxs[x]
        for uop in keys(boundarymap)
            ubs = boundarymap[uop][x]
            hasperiodic = any(
                b -> b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}, ubs
            )
            hastruncating = any(
                b -> b isa AbstractTruncatingBoundary &&
                    !(b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}), ubs
            )
            if spec isa FourierCollocation
                hasperiodic || throw(
                    ArgumentError(
                        "`FourierCollocation` for $x requires a periodic boundary condition `u(t, a) ~ u(t, b)` on every dependent variable in that direction; none was found for $uop."
                    )
                )
                hastruncating && throw(
                    ArgumentError(
                        "Periodic direction $x discretized with `FourierCollocation` does not accept additional truncating boundary conditions like $(first(filter(b -> b isa AbstractTruncatingBoundary && !(b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}), ubs)).eq)."
                    )
                )
            else
                hasperiodic && throw(
                    ArgumentError(
                        "Direction $x has a periodic boundary condition but is discretized with $(spec isa AbstractVector ? "a custom grid" : string(typeof(spec))). Periodic directions require `x => FourierCollocation(n)`."
                    )
                )
            end
        end
    end
    return
end

function PDEBase.should_transform(
        pdesys::PDESystem, disc::PseudospectralDiscretization, boundarymap
    )
    # The collocation evaluator handles nested derivative terms like
    # `Dx(a(u) * Dx(u))` or `Dx(u^2)` natively, so the FD-oriented
    # transformation (auxiliary variables etc.) is unnecessary.
    return false
end

"""
`spectral_clip_interior!!`: higher-order periodic matching conditions are
satisfied identically by the trigonometric interpolant, so they must not
truncate the interior - the corresponding equations are skipped in
`discretize_equation!` as well.
"""
clip_interior!!(lower, upper, s, b, discretization) = clip_interior!!(lower, upper, s, b)
function clip_interior!!(
        lower, upper, s, b::HigherOrderInterfaceBoundary,
        discretization::PseudospectralDiscretization
    )
    return nothing
end

function validate_interface_orders(
        pdes, boundarymap, discretization::PseudospectralDiscretization
    )
    return nothing
end

"""
No stencil extents: spectral rows always span the entire direction, so no
ghost-point padding is ever required.
"""
function calculate_stencil_extents(
        s, u, discretization::PseudospectralDiscretization, orders, bcmap
    )
    n = length(remove(arguments(u), s.time))
    return zeros(Int, n), zeros(Int, n)
end

function PDEBase.construct_discrete_space(
        vars::PDEBase.VariableMap, discretization::PseudospectralDiscretization
    )
    x̄ = vars.x̄
    depvars = vars.ū
    nspace = length(x̄)

    axies = Dict(
        map(x̄) do x
            spec = discretization.dxs[x]
            a, b = vars.intervals[x]
            x => spectral_grid(spec, a, b)
        end
    )
    grid = axies
    dxs = Dict(x => diff(axies[x]) for x in x̄)

    Iaxies = [
        u => CartesianIndices(
            (
                (
                    axes(axies[x])[1]
                        for x in remove(arguments(u), vars.time)
                )...,
            )
        )
            for u in depvars
    ]
    depvarsdisc = discretize_dep_vars(depvars, grid, vars)

    return DiscreteSpace{nspace, length(depvars), CenterAlignedGrid}(
        vars, Dict(depvarsdisc), axies, grid, dxs, Dict(Iaxies), Dict(Iaxies), nothing
    )
end

"""
    SpectralApply{J, S}

Symbolic array operator applying a differentiation operator (a dense block, or a
whole-direction FFT operator) along axis `J` of its argument through
[`apply_along`](@ref), producing an array of size `S`. The operator is a field
rather than a literal in the expression tree, so generated code references it as a
constant instead of spelling out every entry.
"""
struct SpectralApply{J, S, M} <: Function
    mat::M
end
SpectralApply{J, S}(mat::M) where {J, S, M} = SpectralApply{J, S, M}(mat)

(op::SpectralApply{J})(X) where {J} = apply_along(op.mat, X, Val(J))

function SymbolicUtils.promote_symtype(::SpectralApply{J, S}, ::Type) where {J, S}
    return Array{Real, length(S)}
end

function SymbolicUtils.promote_shape(
        ::SpectralApply{J, S}, ::SymbolicUtils.ShapeT
    ) where {J, S}
    return SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, S))
end

"""
`mat` applied along axis `j` of the symbolic array `X`, as a [`SpectralApply`](@ref) term.
"""
function spectral_apply_along(mat, j, X)
    Xu = safe_unwrap(X)
    sz = collect(size(Symbolics.wrap(Xu)))
    sz[j] = mat isa AbstractMatrix ? size(mat, 1) : sz[j]
    S = Tuple(sz)
    sh = SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, S))
    tm = SymbolicUtils.term(
        SpectralApply{j, S}(mat), Xu; type = Array{Real, length(S)}, shape = sh
    )
    return Symbolics.wrap(tm)
end

"""
Whether `ex` varies along `x`: `x` itself appears, or a dependent variable is called
with `x` (rather than a literal) in that slot.
"""
function spectral_depends_on(ex, x, s)
    array_mentions_iv(ex, x) && return true
    xu = safe_unwrap(x)
    for u_ in get_depvars(ex, s.vars.depvar_ops)
        any(a -> isequal(safe_unwrap(a), xu), arguments(u_)) && return true
    end
    return false
end

struct SpectralArrayContext
    s::Any
    derivweights::Any
    indexmap::Any
    args::Any
    depvars::Any
    pde::Any
end

"""
    spectral_arrayify(ex, ranges, c::SpectralArrayContext)

The slice form of `ex` over the box `ranges` (equation axis => index range). Every
outermost spatial-derivative subterm becomes a [`SpectralApply`](@ref) term via
[`spectral_array_derivative`](@ref); everything else is handled by `arrayify` with
the same slice, boundary-value, time-literal and grid rules the finite difference
array form uses.
"""
function spectral_arrayify(ex, ranges, c::SpectralArrayContext)
    s = c.s
    N = length(c.args)
    dterms = Any[]
    find_diff_terms!(dterms, safe_unwrap(ex), s.x̄)
    diffrules = Pair[
        tm => spectral_array_derivative(tm, ranges, c) for tm in unique!(dterms)
    ]
    bvalrules = array_boundary_value_rules(c.pde, s, ranges, c.indexmap, c.derivweights)
    tlrules = array_time_literal_rules(c.pde, s, ranges, c.indexmap, c.derivweights)
    varrules = Pair[]
    for u in c.depvars
        if isempty(ivs(u, s))
            push!(varrules, safe_unwrap(u) => array_scalar_discvar(u, s))
        else
            push!(varrules, safe_unwrap(u) => array_slice(u, s, ranges, c.indexmap))
        end
    end
    gridrules = Pair[
        safe_unwrap(x) => array_grid_vals(x, s, ranges, c.indexmap, N) for x in c.args
    ]
    ctx = ArrayifyContext(
        vcat(diffrules, bvalrules, tlrules, varrules, gridrules), s.time
    )
    return arrayify(ex, ctx)
end

"""
    spectral_array_derivative(term, ranges, c::SpectralArrayContext)

`(Differential(x)^d)(inner)` over `ranges` as the differentiation block for the rows
of `ranges` along `x` applied to `inner` evaluated on the full tap range of that
direction. A dependent variable call with a literal argument pins that axis: a literal
in the `x` slot selects the boundary row (`Dx(u(t, a, y))`), a literal elsewhere
restricts the slice (`Dx(u(t, x, b))`). Terms that do not vary along `x` differentiate
to zero.
"""
function spectral_array_derivative(term, ranges, c::SpectralArrayContext)
    s = c.s
    op = operation(term)
    x = op.x
    inner = only(arguments(term))
    haskey(c.indexmap, x) || throw(
        ArrayFormFallback("derivative in $x, which is not an axis of the equation")
    )
    j = c.indexmap[x]
    D = c.derivweights.map[Differential(x)^op.order]
    innerranges = copy(ranges)
    innerranges[j] = first(D.taps):last(D.taps)
    rows = D.rowmap[ranges[j]]
    val = if iscall(inner) && any(o -> isequal(operation(inner), o), s.vars.depvar_ops)
        u = depvar(inner, s)
        uivs = ivs(u, s)
        any(y -> isequal(y, x), uivs) || return 0
        for (y, a) in zip(uivs, remove(arguments(inner), s.time))
            aval = unwrap_const(safe_unwrap(a))
            aval isa Number || continue
            idx = array_boundary_edge_index(aval, y, s)
            if isequal(y, x)
                rows = D.rowmap[idx:idx]
            else
                innerranges[c.indexmap[y]] = idx:idx
            end
        end
        array_slice(u, s, innerranges, c.indexmap)
    else
        spectral_depends_on(inner, x, s) || return 0
        spectral_arrayify(inner, innerranges, c)
    end
    is_array_valued(val) || return 0
    if D.fast !== nothing && rows == first(rows):last(rows)
        full = spectral_apply_along(D.fast, j, val)
        length(rows) == size(D.mat, 1) && return full
        # differentiate on every node, then keep the rows asked for
        rowrange = first(rows):last(rows)
        rs = ntuple(k -> k == j ? rowrange : (1:size(full, k)), ndims(full))
        return full[rs...]
    end
    return spectral_apply_along(D.mat[rows, :], j, val)
end

"""
    discretize_spectral_array_form(pde, interior, s, depvars, derivweights, eqvar, indexmap)

The interior of `pde` as a single symbolic array equation over the interior box, with
each spatial derivative a dense differentiation block applied along its axis. Throws
`ArrayFormFallback` for patterns without a slice form here: stationary systems and
whatever `arrayify` declines.
"""
function discretize_spectral_array_form(
        pde, interior, s, depvars, derivweights, eqvar, indexmap
    )
    s.time === nothing && throw(
        ArrayFormFallback(
            "stationary (no time) systems have no array form in NonlinearSystem construction"
        )
    )
    args = ivs(eqvar, s)
    N = length(args)
    array_validate_depvar_axes(pde, s, args)
    slicevars = array_sliceable_depvars(
        s, array_unique_depvars(array_pde_occurrences(pde, s), s), args
    )
    lo = Tuple(first(interior))
    hi = Tuple(last(interior))
    length(interior) == prod(hi .- lo .+ 1) ||
        throw(ArrayFormFallback("interior is not a contiguous box"))
    ranges = Dict(j => lo[j]:hi[j] for j in 1:N)
    array_validate_boundary_values(pde, s, derivweights)
    c = SpectralArrayContext(s, derivweights, indexmap, args, slicevars, pde)
    lhs = spectral_arrayify(pde.lhs, ranges, c)
    rhs = spectral_arrayify(pde.rhs, ranges, c)
    donor = array_slice(depvar(eqvar, s), s, ranges, indexmap)
    if is_array_valued(lhs) && !is_array_valued(rhs)
        rhs = array_broadcast_onto(rhs, donor)
    elseif !is_array_valued(lhs) && is_array_valued(rhs)
        lhs = array_broadcast_onto(lhs, donor)
    elseif !is_array_valued(lhs) && !is_array_valued(rhs)
        throw(ArrayFormFallback("equation contains no discretizable terms"))
    end
    return [lhs ~ rhs]
end

"""
`array_boundary_derivative_expr` for a spectral operator: the boundary row of the
differentiation matrix applied to the full slice along `x_`.
"""
function array_boundary_derivative_expr(
        Dop::SpectralDerivativeOperator, II0, s, u, x_, j, N, ranges, indexmap, bcmap
    )
    innerranges = copy(ranges)
    innerranges[j] = first(Dop.taps):last(Dop.taps)
    row = Dop.rowmap[II0[j]]
    return spectral_apply_along(
        Dop.mat[row:row, :], j, array_slice(u, s, innerranges, indexmap)
    )
end

"""
Check that a derivative matching condition across a periodic seam holds identically
under `FourierCollocation`, where both ends of the seam share a differentiation row.
Evaluated at a single edge point: the row selection is the same across the face.
"""
function check_spectral_periodic_matching(s, boundaryvalfuncs, boundary, interiormap)
    u = depvar(boundary.u, s)
    args = ivs(u, s)
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    E = edge(s, boundary, interiormap)
    isempty(E) && return
    II = first(E)
    bc = boundary.eq
    boundaryvalrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs)
    vmaps = varmaps(s, boundary.depvars, II, indexmap)
    varrules = axiesvals(s, u, boundary.x, II)
    rules = Dict(vcat(boundaryvalrules, vmaps, varrules))
    lhs = pde_substitute(bc.lhs, rules)
    rhs = pde_substitute(bc.rhs, rules)
    isequal(lhs, rhs) || throw(
        ArgumentError(
            "Periodic boundary condition $(boundary.eq) is inconsistent with the `FourierCollocation` discretization: derivative matching is automatic for the trigonometric interpolant and cannot express a jump."
        )
    )
    return
end

function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace,
        derivweights::SpectralDifferentialDiscretizer, indexmap,
        discretization::PseudospectralDiscretization
    )
    boundaryvalfuncs = generate_boundary_val_funcs(
        s, depvars, bcmap, indexmap, derivweights
    )
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    for boundary in eqvarbcs
        if boundary isa HigherOrderInterfaceBoundary
            check_spectral_periodic_matching(s, boundaryvalfuncs, boundary, interiormap)
            continue
        end
        try
            vcat!(
                disc_state.bceqs,
                array_bc_eqs(s, boundary, interiormap, derivweights, bcmap)
            )
        catch e
            e isa InterruptException && rethrow(e)
            reason = e isa ArrayFormFallback ? e.msg : sprint(showerror, e)
            @debug "Array form falling back to pointwise boundary equations for $(boundary.eq): $reason"
            generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
        end
    end
    try
        vcat!(
            disc_state.bceqs,
            array_corner_eqs(s, interiormap, eqvar, ndims(s.discvars[eqvar]))
        )
    catch e
        e isa InterruptException && rethrow(e)
        reason = e isa ArrayFormFallback ? e.msg : sprint(showerror, e)
        @debug "Array form falling back to pointwise corner equations: $reason"
        generate_corner_eqs!(
            disc_state, s, interiormap, ndims(s.discvars[eqvar]), eqvar
        )
    end

    interior = interiormap.I[pde]
    eqs = if isempty(interior)
        [
            discretize_equation_at_point(
                CartesianIndex(), s, depvars, pde, derivweights,
                bcmap, eqvar, indexmap, boundaryvalfuncs
            ),
        ]
    else
        try
            discretize_spectral_array_form(
                pde, interior, s, depvars, derivweights, eqvar, indexmap
            )
        catch e
            e isa InterruptException && rethrow(e)
            reason = e isa ArrayFormFallback ? e.msg : sprint(showerror, e)
            @debug "Array form falling back to pointwise discretization for $pde: $reason"
            vec(
                map(interior) do II
                    discretize_equation_at_point(
                        II, s, depvars, pde, derivweights,
                        bcmap, eqvar, indexmap, boundaryvalfuncs
                    )
                end
            )
        end
    end
    return vcat!(disc_state.eqs, eqs)
end
