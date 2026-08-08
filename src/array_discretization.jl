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
# Periodic directions are translation invariant over the whole interior — the scalar path
# never selects a boundary stencil there, it wraps the taps across the seam — so the
# interior is emitted as one array equation per box of the decomposition described in
# `array_bands`, a count that does not depend on the grid resolution.
#
# Interior boundary values (e.g. `u(t, 1)`) map to the matching array element or face
# slice on every array box, including size-1 wrap boxes and frame points (the scalar
# `boundaryvalfuncs` skip interface faces and free-standing corners; this path does not).
# Derivatives of boundary values, time-literal references like `u(0, x)`, and
# edge-aligned grids, fall back.
#
# Nonlinear laplacians `Dx(a(u) * Dx(u))` are emitted in slice form too; see
# `array_nonlinear_laplacian`. Spherical laplacians `r^-2 Dr(r^2 Dr(u))` build on the
# same half-offset machinery; see `array_spherical_diffusion`.
#
# Functional advection schemes (WENO and friends) are traced once on the interior and
# their taps replaced by shifted slices; nonuniform grids and schemes that read the
# grid coordinate still fall back. See `array_function_scheme`.
#
# Staggered grids collapse the same way: each variable's alignment fixes its two stencil
# taps across the interior, so the interior is one array equation per PDE; the (1D,
# single-point) boundaries stay pointwise.
#
# Unsupported patterns fall back to pointwise scalarization (same numerics as
# `ScalarizedDiscretization`): nonuniform functional advection,
# integrals, mixed derivatives, two-variable interfaces, callbacks,
# differing dimensionality, boundary-value derivatives, time-literal
# dependent-variable calls, edge-aligned boundary values, stationary systems.

struct ArrayDiscretizationFallback <: Exception
    msg::String
    # `benign` marks a deliberate choice not to build an array form (nothing would be
    # collapsed) rather than a pattern this path cannot represent. Strict mode tolerates
    # the former and raises on the latter.
    benign::Bool
end
ArrayDiscretizationFallback(msg::String) = ArrayDiscretizationFallback(msg, false)

isbenign(e) = e isa ArrayDiscretizationFallback && e.benign

"""
    ArrayDiscretizationError(pde, msg)

Raised by [`StrictArrayDiscretization`](@ref) when an equation cannot be represented in
slice form, in place of the silent fallback [`ArrayDiscretization`](@ref) performs.
"""
struct ArrayDiscretizationError <: Exception
    pde::Any
    msg::String
end

function Base.showerror(io::IO, e::ArrayDiscretizationError)
    print(
        io,
        """
        StrictArrayDiscretization could not build an array (slice-form) equation for:
          $(e.pde)
        Reason: $(e.msg)

        Use ArrayDiscretization() to discretize this equation pointwise instead, which
        gives the same numerical result without the array representation."""
    )
    return
end

function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
        discretization::MOLFiniteDifference{G, D}
    ) where {G, D <: AnyArrayDiscretization}
    # Boundary handling is identical to the scalarized strategy
    boundaryvalfuncs = generate_boundary_val_funcs(
        s, depvars, bcmap, indexmap, derivweights
    )
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    for boundary in eqvarbcs
        try
            vcat!(
                disc_state.bceqs,
                array_bc_eqs(s, boundary, interiormap, derivweights, bcmap)
            )
        catch e
            e isa InterruptException && rethrow(e)
            reason = e isa ArrayDiscretizationFallback ? e.msg : sprint(showerror, e)
            @debug "ArrayDiscretization falling back to pointwise boundary equations for $(boundary.eq): $reason"
            isstrict(discretization.disc_strategy) && !isbenign(e) &&
                throw(ArrayDiscretizationError(boundary.eq, reason))
            generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
        end
    end
    generate_extrap_eqs!(disc_state, pde, eqvar, s, derivweights, interiormap, bcmap)
    try
        vcat!(
            disc_state.bceqs,
            array_corner_eqs(s, interiormap, eqvar, ndims(s.discvars[eqvar]))
        )
    catch e
        e isa InterruptException && rethrow(e)
        reason = e isa ArrayDiscretizationFallback ? e.msg : sprint(showerror, e)
        @debug "ArrayDiscretization falling back to pointwise corner equations: $reason"
        isstrict(discretization.disc_strategy) && !isbenign(e) &&
            throw(ArrayDiscretizationError(pde, reason))
        generate_corner_eqs!(
            disc_state, s, interiormap, ndims(s.discvars[eqvar]), eqvar
        )
    end

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
            e isa InterruptException && rethrow(e)
            # Any failure to *build* the array form degrades to the pointwise path, which
            # is the reference implementation: this strategy must never turn a system the
            # scalar path can discretize into an error. Genuine problems with the equation
            # itself resurface below, where the scalar path raises them directly.
            reason = e isa ArrayDiscretizationFallback ? e.msg :
                sprint(showerror, e)
            isstrict(discretization.disc_strategy) && !isbenign(e) &&
                throw(ArrayDiscretizationError(pde, reason))
            @debug "ArrayDiscretization falling back to pointwise discretization for $pde: $reason"
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

Discretize the interior of `pde` as symbolic array equations over slices of the
discretized dependent variables — one per box of the decomposition built by
`array_bands`, which is a single box unless the equation has periodic directions — plus
pointwise scalar equations for any interior points whose stencils differ from the
translation-invariant interior stencil. Throws `ArrayDiscretizationFallback` when `pde`
contains a pattern that cannot be represented this way.
"""
function discretize_equation_array_form(
        pde, interior, s, depvars, derivweights, bcmap,
        eqvar, indexmap, boundaryvalfuncs
    )
    isstag = get_grid_type(s) <: StaggeredGrid
    # Stationary: PDEBase emits `0 ~ residual`; Symbolics rejects scalar ~ array.
    s.time === nothing && throw(
        ArrayDiscretizationFallback(
            "stationary (no time) systems have no array form in NonlinearSystem construction"
        )
    )
    # The staggered path never consults the advection scheme (see the staggered
    # `generate_finite_difference_rules`), so the requirement only applies otherwise.
    isstag || derivweights.advection_scheme isa Union{UpwindScheme, FunctionalScheme} ||
        throw(
        ArrayDiscretizationFallback(
            "unsupported advection scheme $(derivweights.advection_scheme)"
        )
    )

    args = ivs(eqvar, s)
    for u in depvars
        isequal(ivs(u, s), args) ||
            throw(ArrayDiscretizationFallback("variables of differing dimensionality"))
    end
    periodic = array_periodic_dims(s, depvars, args, bcmap)

    pdeorders = Dict(x => d_orders(x, [pde]) for x in args)
    isstag && validate_staggered_array_form(s, depvars, pdeorders, args)
    terms = split_terms(pde, s.x̄)
    # Matched before the bands: half-offset stencils reach further than the order-2
    # central difference `d_orders` reports for these terms. The staggered path has no
    # nonlinear-laplacian or spherical scheme; `validate_staggered_array_form` already
    # rejects even orders, so there is nothing to match there.
    nllap_matches = NonlinlapMatch[]
    nllap_orders = Dict()
    sph_matches = SphericalMatch[]
    sph_orders = Dict()
    if !isstag
        nllap_matches = match_nonlinlap_terms(terms, s, depvars)
        for m in nllap_matches
            orders = Set(nonlinlap_coeff_orders(m, depvars, derivweights))
            nllap_orders[m.x] = union(get(nllap_orders, m.x, Set{Int}()), orders)
        end
        sph_matches = match_spherical_terms(
            split_additive_terms(pde), s, depvars, nllap_matches
        )
        for m in sph_matches
            orders = Set(nonlinlap_coeff_orders(m, depvars, derivweights))
            sph_orders[m.x] = union(get(sph_orders, m.x, Set{Int}()), orders)
        end
    end
    bands, clean = array_bands(
        interior, s, args, pdeorders, derivweights, indexmap, periodic, nllap_orders,
        sph_orders
    )
    any(isempty, bands) && throw(ArrayDiscretizationFallback("empty core region"))
    N = length(args)
    core = CartesianIndices(
        ntuple(j -> first(first(bands[j])):last(last(bands[j])), N)
    )

    # Probe the special-case rulesets at a representative core point, one whose stencils
    # do not wrap. Several of these generators return candidate rules unconditionally; the
    # scalar path only applies a special scheme when a rule key occurs in the equation, so
    # fall back exactly when one does. Any firing rule means a scheme with no slice
    # representation here yet. The staggered scalar path applies none of these schemes,
    # so there is nothing to probe there; its unsupported patterns (integrals, ...)
    # surface in `arrayify` instead.
    if !isstag
        II0 = CartesianIndex(
            ntuple(j -> clean[j] === nothing ? first(core)[j] : first(bands[j][clean[j]]), N)
        )
        special_rules = vcat(
            vec(generate_mixed_rules(II0, s, depvars, derivweights, bcmap, indexmap, terms)),
            vec(generate_euler_integration_rules(II0, s, depvars, indexmap, terms)),
            vec(generate_whole_domain_integration_rules(II0, s, depvars, indexmap, terms)),
            vec(generate_cb_rules(II0, s, depvars, derivweights, bcmap, indexmap, terms))
        )
        for r in special_rules
            (subsmatch(pde.lhs, r) || subsmatch(pde.rhs, r)) &&
                throw(ArrayDiscretizationFallback("unsupported pattern $(r.first)"))
        end
    end
    array_validate_boundary_values(pde, s)

    core_eqs = map(Iterators.product(map(eachindex, bands)...)) do combo
        rs = ntuple(j -> bands[j][combo[j]], N)
        ranges = Dict(j => rs[j] for j in 1:N)
        if prod(map(length, rs)) == 1
            # a single point is more clearly written as the scalar equation it is;
            # still apply array bval rules (periodic faces have empty boundaryvalfuncs)
            eq = discretize_equation_at_point(
                CartesianIndex(map(first, rs)), s, depvars, pde, derivweights,
                bcmap, eqvar, indexmap, boundaryvalfuncs
            )
            return array_substitute_boundary_values(eq, pde, s, ranges, indexmap)
        end
        return array_core_equation(
            pde, ranges, s, depvars, derivweights,
            args, pdeorders, indexmap, terms, periodic, nllap_matches, sph_matches
        )
    end

    frame = setdiff(vec(collect(interior)), vec(collect(core)))
    frame_eqs = map(frame) do II
        eq = discretize_equation_at_point(
            II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap, boundaryvalfuncs
        )
        ranges = Dict(j => II[j]:II[j] for j in 1:N)
        return array_substitute_boundary_values(eq, pde, s, ranges, indexmap)
    end
    return vcat(vec(core_eqs), frame_eqs)
end

"""
The array equation for one box of the interior, given as `ranges` (dimension => index
range).
"""
function array_core_equation(
        pde, ranges, s, depvars, derivweights, args, pdeorders, indexmap, terms,
        periodic, nllap_matches, sph_matches
    )
    N = length(args)
    shape = ntuple(j -> length(ranges[j]), N)
    # First matching rule wins. Boundary-value rules before core-variable rules.
    bvalrules = array_boundary_value_rules(pde, s, ranges, indexmap)
    varrules = [safe_unwrap(u) => array_slice(u, s, ranges, indexmap) for u in depvars]
    gridrules = [
        safe_unwrap(x) => array_grid_vals(x, s, ranges, indexmap, N) for x in args
    ]
    if get_grid_type(s) <: StaggeredGrid
        derivrules = array_staggered_rules(
            s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
        )
        windrules = Pair[]
        nllaprules = Pair[]
        sphrules = Pair[]
        advrules = Pair[]
    else
        derivrules = array_cartesian_rules(
            s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
        )
        # Winding coefficients are arrayified with the same baserules; include bvalrules so
        # e.g. `u(t, 1)*Dx(u)` substitutes the boundary element before the wind rule fires.
        windrules = array_winding_rules(
            terms, s, depvars, pdeorders, derivweights, ranges, indexmap,
            vcat(bvalrules, varrules, gridrules), periodic
        )
        nllaprules = array_nonlinlap_rules(
            nllap_matches, s, depvars, derivweights, ranges, indexmap,
            vcat(bvalrules, varrules, gridrules), periodic
        )
        sphrules = array_spherical_rules(
            sph_matches, s, depvars, derivweights, ranges, indexmap,
            vcat(bvalrules, varrules, gridrules), periodic
        )
        advrules = array_advection_rules(
            s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
        )
    end
    # Family order is the reverse of the scalar path's last-key-wins `Dict`.
    ctx = ArrayifyContext(
        vcat(bvalrules, windrules, advrules, nllaprules, sphrules, derivrules, varrules, gridrules),
        s.time
    )

    lhs = arrayify(pde.lhs, ctx)
    rhs = arrayify(pde.rhs, ctx)
    is_zero_scalar(x) =
    let v = unwrap_const(safe_unwrap(x))
        v isa Number && iszero(v)
    end
    # `~` cannot equate an array with a scalar; the system is cardinalized so the rhs is
    # (a scalar) 0 and the lhs holds the whole residual.
    if is_array_valued(lhs) && !is_array_valued(rhs)
        is_zero_scalar(rhs) ||
            throw(ArrayDiscretizationFallback("array lhs with non-zero scalar rhs"))
        rhs = zeros(shape)
    elseif !is_array_valued(lhs) && is_array_valued(rhs)
        is_zero_scalar(lhs) ||
            throw(ArrayDiscretizationFallback("array rhs with non-zero scalar lhs"))
        lhs = zeros(shape)
    elseif !is_array_valued(lhs) && !is_array_valued(rhs)
        throw(ArrayDiscretizationFallback("equation contains no discretizable terms"))
    end
    return lhs ~ rhs
end

"""
The directions in which every dependent variable is periodic, mapped to the number of grid
points in that direction.

A periodic direction is one whose interface boundaries join a variable to itself at the
other end of the same independent variable, which is what `u(t, 0) ~ u(t, 1)` parses to.
There the scalar path never selects a boundary stencil — `haslowerupper` reports both ends
as interfaces — so the interior stencil applies across the whole interior with its taps
wrapped around the seam by `bwrap`, which `wrap_periodic_range` reproduces on slices.

Interfaces that join two different variables, or one end of a domain only, shift taps onto
another array and have no such form here; those throw.
"""
function array_periodic_dims(s, depvars, args, bcmap)
    periodic = Dict()
    for x in args
        withiface = filter(u -> !isempty(filter_interfaces(bcmap[operation(u)][x])), depvars)
        isempty(withiface) && continue
        length(withiface) == length(depvars) ||
            throw(ArrayDiscretizationFallback("interface boundaries on only some variables in $x"))
        for u in withiface
            bs = filter_interfaces(bcmap[operation(u)][x])
            all(haslowerupper(bs, x)) ||
                throw(ArrayDiscretizationFallback("interface boundary at one end of $x only"))
            for b in bs
                isequal(b.x, x) && isequal(b.x2, x) &&
                    isequal(depvar(b.u, s), depvar(u, s)) &&
                    isequal(depvar(b.u2, s), depvar(u, s)) ||
                    throw(
                    ArrayDiscretizationFallback(
                        "interface boundary $(b.eq) joins different variables"
                    )
                )
            end
        end
        # `central_difference_weights_and_stencil` and `_upwind_difference` reject
        # interfaces on a nonuniform grid outright, so the scalar path is the one that
        # must report that.
        s.dxs[x] isa Number ||
            throw(ArrayDiscretizationFallback("interface boundary on a nonuniform grid"))
        periodic[x] = length(s, x)
    end
    return periodic
end

"""
Whether the derivative of order `d` is discretized by a functional advection scheme
(WENO and friends) rather than by the winding rules, mirroring the branch on the advection
scheme in `generate_finite_difference_rules`: only the first derivative is handled by the
scheme, the higher odd orders still wind.
"""
array_functional_advection(derivweights, d) =
    d == 1 && derivweights.advection_scheme isa FunctionalScheme

"""
The most negative and most positive tap offsets any derivative in the equation applies in
direction `x`, mirroring the interior branches of `central_difference_weights_and_stencil`,
`_upwind_difference` and `get_f_taps_coords`.

On a staggered grid the interior taps are `(0, +1)` or `(-1, 0)` depending on each
variable's alignment; the union over both alignments is taken, so at worst one extra
point per end lands in the pointwise frame.
"""
function array_tap_extents(x, pdeorders, derivweights, ::Type{G}) where {G}
    mintap = 0
    maxtap = 0
    for d in pdeorders[x]
        if iseven(d)
            Dop = derivweights.map[Differential(x)^d]
            taps = half_range(Dop.stencil_length)
            mintap = min(mintap, first(taps))
            maxtap = max(maxtap, last(taps))
        elseif array_functional_advection(derivweights, d)
            taps = half_range(derivweights.advection_scheme.interior_points)
            mintap = min(mintap, first(taps))
            maxtap = max(maxtap, last(taps))
        else
            Dneg = derivweights.windmap[1][Differential(x)^d]
            Dpos = derivweights.windmap[2][Differential(x)^d]
            mintap = min(mintap, -Dpos.stencil_length + 1)
            maxtap = max(maxtap, Dneg.stencil_length - 1)
        end
    end
    return mintap, maxtap
end

function array_tap_extents(x, pdeorders, derivweights, ::Type{G}) where {G <: StaggeredGrid}
    return isempty(pdeorders[x]) ? (0, 0) : (-1, 1)
end

"""
Decompose the interior into the boxes on which one array equation is valid, as the index
ranges to take in each dimension (the boxes are their cartesian product). Returns those
ranges and, per dimension, which of them holds points whose stencils do not wrap.

In a direction with no interface boundaries this is the single subbox of the interior on
which every derivative resolves to the translation-invariant interior stencil, mirroring
the branch conditions in `central_difference_weights_and_stencil` and
`_upwind_difference`; interior points outside it — the frame — are discretized pointwise.

A periodic direction has no such boundary branch: the interior stencil applies across the
whole interior, but for points within a stencil of either end some taps wrap around the
seam and the wrapped tap is not contiguous with the rest. Splitting those points off as
one range each keeps every tap of every box a single contiguous slice, at the cost of a
handful of extra equations — as many as the stencil is wide, so the count still does not
depend on the grid resolution.

`nllap_orders` maps each direction carrying a nonlinear laplacian to the coefficient's
derivative orders above one; those directions additionally take the half-offset branch
conditions and tap extents of `array_nonlinlap_constraints`. `sph_orders` does the same
for spherical laplacians via `array_spherical_constraints`, and additionally keeps any
r ≈ 0 points out of the core (the scalar path treats them with a separate branch).
"""
function array_bands(
        interior, s, args, pdeorders, derivweights, indexmap, periodic,
        nllap_orders = Dict(), sph_orders = Dict()
    )
    N = length(args)
    bands = [UnitRange{Int}[] for _ in 1:N]
    clean = Vector{Union{Nothing, Int}}(nothing, N)
    for x in args
        j = indexmap[x]
        lo = first(interior)[j]
        hi = last(interior)[j]
        n = length(s, x)
        mintap, maxtap = array_tap_extents(x, pdeorders, derivweights, get_grid_type(s))
        nl = if haskey(nllap_orders, x)
            c = array_nonlinlap_constraints(
                x, n, derivweights, sort(collect(nllap_orders[x]))
            )
            mintap = min(mintap, c[3])
            maxtap = max(maxtap, c[4])
            c
        else
            nothing
        end
        sph = if haskey(sph_orders, x)
            haskey(periodic, x) && throw(
                ArrayDiscretizationFallback("spherical laplacian in a periodic direction")
            )
            c = array_spherical_constraints(
                x, n, derivweights, sort(collect(sph_orders[x]))
            )
            mintap = min(mintap, c[3])
            maxtap = max(maxtap, c[4])
            c
        else
            nothing
        end
        if !haskey(periodic, x)
            # Taps must stay in range, and no point may take a boundary branch: the
            # centered one at II <= boundary_point_count or II > n - boundary_point_count,
            # the positive winding at II <= offside, the negative one at
            # II > n - boundary_point_count. The staggered branch conditions use the
            # centered operator's boundary_point_count for every order.
            lo = max(lo, 1 - mintap)
            hi = min(hi, n - maxtap)
            for d in pdeorders[x]
                if get_grid_type(s) <: StaggeredGrid
                    bpc = derivweights.map[Differential(x)^d].boundary_point_count
                    lo = max(lo, bpc + 1)
                    hi = min(hi, n - bpc)
                elseif iseven(d)
                    bpc = derivweights.map[Differential(x)^d].boundary_point_count
                    lo = max(lo, bpc + 1)
                    hi = min(hi, n - bpc)
                elseif array_functional_advection(derivweights, d)
                    # the branch conditions of `get_f_taps_coords`: points within
                    # `length(F.lower)` of the start, or `length(F.upper)` of the end, take
                    # a boundary function of the scheme instead of its interior one
                    F = derivweights.advection_scheme
                    lo = max(lo, length(F.lower) + 1)
                    hi = min(hi, n - length(F.upper))
                else
                    lo = max(lo, derivweights.windmap[2][Differential(x)^d].offside + 1)
                    hi = min(
                        hi,
                        n - derivweights.windmap[1][Differential(x)^d].boundary_point_count
                    )
                end
            end
            if nl !== nothing
                lo = max(lo, nl[1])
                hi = min(hi, nl[2])
            end
            if sph !== nothing
                lo = max(lo, sph[1])
                hi = min(hi, sph[2])
                # r ≈ 0 takes a separate scalar branch (appendix B), keep it in the frame.
                grid = s.grid[x]
                while lo <= hi && abs(grid[lo]) <= 1.0e-6
                    lo += 1
                end
                while lo <= hi && abs(grid[hi]) <= 1.0e-6
                    hi -= 1
                end
                any(i -> abs(grid[i]) <= 1.0e-6, lo:hi) && throw(
                    ArrayDiscretizationFallback(
                        "spherical laplacian with r ≈ 0 inside the core"
                    )
                )
            end
            lo > hi && return bands, clean
            bands[j] = [lo:hi]
            clean[j] = 1
            continue
        end
        wraps(i) = (i + mintap <= 1) || (i + maxtap > n)
        lastlow = lo - 1
        while lastlow < hi && wraps(lastlow + 1)
            lastlow += 1
        end
        firsthigh = hi + 1
        while firsthigh > lastlow + 1 && wraps(firsthigh - 1)
            firsthigh -= 1
        end
        bands[j] = [i:i for i in lo:lastlow]
        if lastlow < firsthigh - 1
            push!(bands[j], (lastlow + 1):(firsthigh - 1))
            clean[j] = length(bands[j])
        end
        append!(bands[j], [i:i for i in firsthigh:hi])
    end
    return bands, clean
end

"""
The index range a tap slice takes across a periodic seam, mirroring `_wrapinterface`:
indices at or below the first point come from the far end of the grid, indices past the
last point from its start. A range that straddles the seam is not a slice.
"""
function wrap_periodic_range(r, n)
    lo, hi = first(r), last(r)
    lo > n && return (lo - n + 1):(hi - n + 1)
    hi <= 1 && return (lo + n - 1):(hi + n - 1)
    (lo >= 2 && hi <= n) && return r
    throw(ArrayDiscretizationFallback("periodic stencil tap straddles the seam"))
end

"""
The underlying (unscalarized) array variable of which `s.discvars[u]` holds the elements.
"""
function array_variable(u, s)
    el = safe_unwrap(first(vec(s.discvars[u])))
    (iscall(el) && operation(el) === getindex) ||
        throw(ArrayDiscretizationFallback("discrete variable for $u is not an array variable"))
    arr = first(arguments(el))
    # For an array-valued dependent variable (`@variables u(..)[1:n]`) the discrete
    # variable is a nested getindex, so the immediate parent is a component rather than
    # the grid-shaped array this path can slice.
    T = SymbolicUtils.symtype(arr)
    (T <: AbstractArray && ndims(T) == length(ivs(u, s))) ||
        throw(ArrayDiscretizationFallback("discrete variable for $u is not a grid-shaped array"))
    return Symbolics.wrap(arr)
end

"""
A slice of the array variable for `u` over the core region, optionally shifted by
`offset` in the dimension of `shiftx`, wrapped around the seam if that dimension is
periodic (see `wrap_periodic_range`).
"""
function array_slice(u, s, ranges, indexmap; shiftx = nothing, offset = 0, periodic = nothing)
    arr = array_variable(depvar(u, s), s)
    rs = map(ivs(depvar(u, s), s)) do y
        r = ranges[indexmap[y]]
        (shiftx !== nothing && isequal(y, shiftx)) || return r
        r = r .+ offset
        return (periodic !== nothing && haskey(periodic, y)) ?
            wrap_periodic_range(r, periodic[y]) : r
    end
    return arr[rs...]
end

# Boundary values in interior equations

"""
Resolve a numeric boundary argument to a 1-based grid index at a domain edge,
matching the arithmetic in `newindex`.
"""
function array_boundary_edge_index(xval, x, s)
    if isequal(xval, s.axies[x][1])
        return 1
    elseif isequal(xval, s.axies[x][end])
        return length(s, x)
    else
        throw(
            ArrayDiscretizationFallback(
                "boundary value is not at a domain edge for $x = $xval"
            )
        )
    end
end

"""
True when `expr` contains a spatial derivative of a boundary value such as
`(Differential(x))(u(t, 1))`. Those have no slice form here yet (the scalar path
handles them via `depvarderivbcmaps`).
"""
function array_has_boundary_value_derivative(expr, s)
    expr = safe_unwrap(expr)
    iscall(expr) || return false
    op = operation(expr)
    if op isa Differential && (s.time === nothing || !isequal(op.x, s.time))
        for a in arguments(expr)
            for v in get_depvars(a, s.vars.depvar_ops)
                any(x -> unwrap_const(safe_unwrap(x)) isa Number, arguments(v)) &&
                    return true
            end
            array_has_boundary_value_derivative(a, s) && return true
        end
    end
    return any(a -> array_has_boundary_value_derivative(a, s), arguments(expr))
end

"""
Dependent-variable terms in `pde` that carry at least one numeric (boundary) argument.
"""
function array_boundary_value_terms(pde, s)
    terms = []
    for u in get_depvars(pde.lhs, s.vars.depvar_ops) ∪ get_depvars(pde.rhs, s.vars.depvar_ops)
        any(x -> unwrap_const(safe_unwrap(x)) isa Number, arguments(u)) && push!(terms, u)
    end
    return terms
end

"""
True when `u_` replaces the time variable with a numeric literal, e.g. `u(0, x)`.
Those are not spatial boundary values; there is no slice form for them here yet.
"""
function array_is_time_literal_term(u_, s)
    s.time === nothing && return false
    args = arguments(u_)
    any(a -> isequal(a, s.time), args) && return false
    return any(a -> unwrap_const(safe_unwrap(a)) isa Number, args)
end

"""
Equation-level checks for boundary values in the interior equation. Throws
`ArrayDiscretizationFallback` for patterns with no slice form yet (edge-aligned grids,
derivatives of boundary values, time literals, off-edge sampling); otherwise succeeds so
`array_boundary_value_rules` can substitute each term.
"""
function array_validate_boundary_values(pde, s)
    bvals = array_boundary_value_terms(pde, s)
    isempty(bvals) && return bvals
    get_grid_type(s) <: CenterAlignedGrid || throw(
        ArrayDiscretizationFallback(
            "boundary values in interior equations require a CenterAlignedGrid"
        )
    )
    (
        array_has_boundary_value_derivative(pde.lhs, s) ||
            array_has_boundary_value_derivative(pde.rhs, s)
    ) && throw(
        ArrayDiscretizationFallback("derivative of boundary value in interior equation")
    )
    for u_ in bvals
        array_is_time_literal_term(u_, s) && throw(
            ArrayDiscretizationFallback(
                "time-literal value $u_ in interior equation (not a spatial boundary value)"
            )
        )
        u = depvar(u_, s)
        args = ivs(u, s)
        args_ = remove(arguments(u_), s.time)
        length(args_) == length(args) || throw(
            ArrayDiscretizationFallback(
                "boundary value $u_ has unexpected argument structure"
            )
        )
        for (j, a) in enumerate(args_)
            aval = unwrap_const(safe_unwrap(a))
            aval isa Number || continue
            array_boundary_edge_index(aval, args[j], s)
        end
    end
    return bvals
end

"""
The array element (every spatial argument fixed) or face slice (some arguments free
over the core box) corresponding to a boundary value like `u(t, 1)` or `u(t, 0, y)`.
Fixed dimensions use a singleton `k:k` range so the result broadcasts against the
core slice; a fully-fixed reference is a scalar element, which broadcasts as well.
"""
function array_boundary_value_view(u_, s, ranges, indexmap)
    u = depvar(u_, s)
    arr = array_variable(u, s)
    args = ivs(u, s)
    args_ = remove(arguments(u_), s.time)
    idxs = Vector{Any}(undef, length(args_))
    all_fixed = true
    for (j, a) in enumerate(args_)
        aval = unwrap_const(safe_unwrap(a))
        if aval isa Number
            idxs[j] = array_boundary_edge_index(aval, args[j], s)
        else
            all_fixed = false
            idxs[j] = ranges[indexmap[args[j]]]
        end
    end
    all_fixed && return arr[idxs...]
    rs = ntuple(j -> idxs[j] isa Integer ? (idxs[j]:idxs[j]) : idxs[j], length(idxs))
    return arr[rs...]
end

"""
Substitution rules mapping each boundary value in `pde` to its array element or face
slice over the core box described by `ranges`. Call after
`array_validate_boundary_values`.

Returns a `Vector{<:Pair}` even when empty, so `vcat` into `ArrayifyContext.rules`
stays well-typed.
"""
function array_boundary_value_rules(pde, s, ranges, indexmap)
    return Pair[
        safe_unwrap(u_) => array_boundary_value_view(u_, s, ranges, indexmap)
            for u_ in array_boundary_value_terms(pde, s)
    ]
end

"""
Substitute boundary values into an already pointwise-discretized equation at the
single point described by `ranges` (all singleton). Used for size-1 wrap boxes and
frame points, where `discretize_equation_at_point` leaves interface-face and
free-standing-corner boundary values symbolic. `valmaps` has already replaced free
spatial arguments with their grid values inside those leftover terms (`u(t, 0, y)`
appears as e.g. `u(t, 0, 0.2)`), so each rule keys on that grid-valued form and maps
to the scalar array element at this point, never a slice.
"""
function array_substitute_boundary_values(eq, pde, s, ranges, indexmap)
    bvals = array_boundary_value_terms(pde, s)
    isempty(bvals) && return eq
    rdict = Dict()
    for u_ in bvals
        u = depvar(u_, s)
        args = ivs(u, s)
        args_ = remove(arguments(u_), s.time)
        idxs = map(enumerate(args_)) do (j, a)
            aval = unwrap_const(safe_unwrap(a))
            aval isa Number ? array_boundary_edge_index(aval, args[j], s) :
                first(ranges[indexmap[args[j]]])
        end
        gridsubs = Dict(
            safe_unwrap(args[j]) => s.grid[args[j]][idxs[j]]
                for (j, a) in enumerate(args_)
                if !(unwrap_const(safe_unwrap(a)) isa Number)
        )
        rdict[pde_substitute(u_, gridsubs)] = array_variable(u, s)[idxs...]
    end
    return pde_substitute(eq.lhs, rdict) ~ pde_substitute(eq.rhs, rdict)
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
    wterms = [broadcast(*, w, sl) for (w, sl) in zip(weights, slices)]
    length(wterms) == 1 && return wterms[1]
    return broadcast(+, wterms...)
end

"""
Array form of `central_difference` on the core region for the even order derivative
`(Differential(x)^d)(u)`.
"""
function array_central_difference(Dop, s, u, x, d, ranges, indexmap, periodic)
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]
    taps = half_range(Dop.stencil_length)
    slices = [
        array_slice(u, s, ranges, indexmap; shiftx = x, offset = k, periodic = periodic)
            for k in taps
    ]
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
        s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
    )
    rules = Pair[]
    for u in depvars, x in ivs(depvar(u, s), s)
        for d in filter(iseven, pdeorders[x])
            Dop = derivweights.map[Differential(x)^d]
            push!(
                rules,
                safe_unwrap((Differential(x)^d)(u)) => array_central_difference(
                    Dop, s, u, x, d, ranges, indexmap, periodic
                )
            )
        end
    end
    return rules
end

"""
Patterns the staggered scalar path cannot discretize either; falling back keeps this
strategy's behaviour identical to `ScalarizedDiscretization` for them.
"""
function validate_staggered_array_form(s, depvars, pdeorders, args)
    for x in args
        all(isodd, pdeorders[x]) ||
            throw(ArrayDiscretizationFallback("even-order derivative on a staggered grid"))
        isempty(pdeorders[x]) && continue
        # the staggered scalar path applies `stencil_coefs` directly, which only holds a
        # single weight set on a uniform grid
        s.dxs[x] isa Number ||
            throw(ArrayDiscretizationFallback("staggered grid with nonuniform d$x"))
    end
    for u in depvars
        haskey(s.staggeredvars, operation(depvar(u, s))) ||
            throw(ArrayDiscretizationFallback("no alignment recorded for $u"))
    end
    return nothing
end

"""
Array form of the interior branch of the staggered `generate_cartesian_rules`: the same
`windmap` weights, with the two taps fixed by each variable's alignment — `(0, +1)` for a
center-aligned variable, `(-1, 0)` for an edge-aligned one — constant across the core.
"""
function array_staggered_rules(
        s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
    )
    rules = Pair[]
    for u in depvars, x in ivs(depvar(u, s), s)
        for d in filter(isodd, pdeorders[x])
            Dop = get(derivweights.windmap[1], Differential(x)^d, nothing)
            Dop === nothing && throw(
                ArrayDiscretizationFallback(
                    "no upwind operator for order $d in $x on a staggered grid"
                )
            )
            taps = s.staggeredvars[operation(depvar(u, s))] == CenterAlignedVar ?
                (0:1) : (-1:0)
            slices = [
                array_slice(
                        u, s, ranges, indexmap;
                        shiftx = x, offset = k, periodic = periodic
                    ) for k in taps
            ]
            push!(
                rules,
                safe_unwrap((Differential(x)^d)(u)) =>
                    array_stencil(collect(Dop.stencil_coefs), slices)
            )
        end
    end
    return rules
end

"""
Array form of `upwind_difference` on the core region, for one winding direction.
"""
function array_upwind_difference(
        s, u, x, d, derivweights, ranges, indexmap, ispositive, periodic
    )
    Dop = ispositive ? derivweights.windmap[2][Differential(x)^d] :
        derivweights.windmap[1][Differential(x)^d]
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]
    taps = ispositive ? ((-Dop.stencil_length + 1):0) : (0:(Dop.stencil_length - 1))
    slices = [
        array_slice(u, s, ranges, indexmap; shiftx = x, offset = k, periodic = periodic)
            for k in taps
    ]
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
`expr`, mirroring the scalar path's `ifelse(coef > 0, coef*pos, coef*neg)`.

When the coefficient does not vary over the grid — a literal, a parameter, or any
expression of time alone — the wind direction is one scalar condition for the whole
slice, so `ifelse` broadcasts and reproduces the scalar path exactly.

A grid-varying coefficient needs a per-point condition, and `ifelse` cannot be broadcast
over a symbolic array condition (the elementwise comparison carries symtype `Any` rather
than `Bool`). Those use `max(coef, 0)*pos + min(coef, 0)*neg`, which agrees with `ifelse`
on finite values but yields `NaN` rather than the finite branch when the unselected
stencil is `Inf`/`NaN`.
"""
function array_winding_select(
        expr, s, u, x, d, derivweights, ranges, indexmap, coefctx, periodic
    )
    coef = arrayify(expr, coefctx)
    pos = array_upwind_difference(
        s, u, x, d, derivweights, ranges, indexmap, true, periodic
    )
    neg = array_upwind_difference(
        s, u, x, d, derivweights, ranges, indexmap, false, periodic
    )
    bcast(op, args...) = broadcast(op, args...)
    if !is_array_valued(coef)
        return bcast(ifelse, coef > 0, bcast(*, coef, pos), bcast(*, coef, neg))
    end
    return bcast(
        +,
        bcast(*, bcast(max, coef, 0), pos),
        bcast(*, bcast(min, coef, 0), neg)
    )
end

"""
Placeholder symbols standing in for one argument vector of a functional scheme while it is
traced. They are substituted away before the expression leaves `array_function_scheme`, so
they never reach the discretized system; the `##` prefix keeps them clear of user names.
"""
array_scheme_syms(tag, n) = [Symbolics.variable(Symbol("##mol_scheme_", tag), i) for i in 1:n]

"""
Array form of `function_scheme` on the interior branch, for the first derivative of `u` in
`x` under a functional advection scheme (WENO and friends).

The scheme is translation invariant on the interior — the same function applied to the same
tap offsets at every point — so it can be traced once on placeholder symbols and its taps
then replaced by shifted slices, instead of being retraced at every grid point. That the
scheme's weights are solution dependent, as WENO's smoothness indicators are, costs nothing
here: they are arithmetic on the taps like any other term, so `arrayify` broadcasts them
elementwise along with the rest of the expression.

The scheme's independent variable argument takes a different value at each point of the
slice, so a scheme that uses it would have to be traced symbolically in that argument too,
and the arithmetic Julia folds over those (numeric) coordinates in the scalar path would be
rebuilt as symbolic operations, in general reassociated. Those fall back rather than risk
differing from the scalar path in the last digits, as do nonuniform grids, whose stepsize
argument varies from point to point in the same way.
"""
function array_function_scheme(F, s, u, x, ranges, indexmap, periodic)
    dx = s.dxs[x]
    dx isa Number ||
        throw(ArrayDiscretizationFallback("$(F.name) advection on a nonuniform grid"))
    # `get_f_taps_coords` rejects a stencil that wraps more than once around the seam
    (haskey(periodic, x) && periodic[x] - 1 < F.interior_points) &&
        throw(ArrayDiscretizationFallback("too few points in $x for $(F.name) to wrap"))
    taps = half_range(F.interior_points)
    usyms = array_scheme_syms("u", length(taps))
    xsyms = array_scheme_syms("x", length(taps))
    expr = try
        F.interior(usyms, vcat(F.ps, params(s)), Num(s.time), xsyms, dx)
    catch e
        e isa InterruptException && rethrow(e)
        throw(ArrayDiscretizationFallback("could not evaluate scheme $(F.name)"))
    end
    any(
        v -> any(y -> isequal(v, safe_unwrap(y)), xsyms),
        Symbolics.get_variables(expr)
    ) && throw(
        ArrayDiscretizationFallback("scheme $(F.name) depends on the grid coordinate")
    )
    slices = [
        array_slice(u, s, ranges, indexmap; shiftx = x, offset = k, periodic = periodic)
            for k in taps
    ]
    rules = [safe_unwrap(usyms[k]) => slices[k] for k in eachindex(taps)]
    return arrayify(expr, ArrayifyContext(rules, s.time))
end

@inline function array_advection_rules(
        s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
    )
    F = derivweights.advection_scheme
    F isa FunctionalScheme || return Pair[]
    rules = Pair[]
    for u in depvars, x in ivs(depvar(u, s), s)
        1 in pdeorders[x] || continue
        push!(
            rules,
            safe_unwrap(Differential(x)(u)) => array_function_scheme(
                F, s, u, x, ranges, indexmap, periodic
            )
        )
    end
    return rules
end

@inline function array_winding_rules(
        terms, s, depvars, pdeorders, derivweights, ranges, indexmap, baserules, periodic
    )
    coefctx = ArrayifyContext(baserules, s.time)
    ruleobjs = []
    for u in depvars, x in ivs(depvar(u, s), s)
        for d in filter(d -> isodd(d) && !array_functional_advection(derivweights, d), pdeorders[x])
            push!(
                ruleobjs,
                @rule *(
                    ~~a, $(Differential(x)^d)(u), ~~b
                ) => array_winding_select(
                    *(~a..., ~b...), s, u, x, d, derivweights,
                    ranges, indexmap, coefctx, periodic
                )
            )
            push!(
                ruleobjs,
                @rule /(
                    *(~~a, $(Differential(x)^d)(u), ~~b), ~c
                ) => array_winding_select(
                    *(~a..., ~b...) / ~c, s, u, x, d, derivweights,
                    ranges, indexmap, coefctx, periodic
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
        for d in filter(d -> isodd(d) && !array_functional_advection(derivweights, d), pdeorders[x])
            push!(
                windrules,
                safe_unwrap((Differential(x)^d)(u)) => array_upwind_difference(
                    s, u, x, d, derivweights, ranges, indexmap, true, periodic
                )
            )
        end
    end
    return windrules
end

####
# Nonlinear laplacian in slice form
####

"""
A matched nonlinear laplacian `Dx(expr * Dx(u))`: `term` is the matched additive term,
`pre` any prefactor product and `div` any divisor (`nothing` when absent).
"""
struct NonlinlapMatch
    term::Any
    x::Any
    u::Any
    expr::Any
    pre::Any
    div::Any
end

_nllap_pre(preargs) = isempty(preargs) ? nothing : *(preargs...)
function _nllap_match(x, u, exprargs, preargs, div)
    return NonlinlapMatch(nothing, x, u, *(exprargs...), _nllap_pre(preargs), div)
end
function _nllap_match_div(x, u, a, preargs, div)
    return NonlinlapMatch(nothing, x, u, 1 / a, _nllap_pre(preargs), div)
end

"""
Match the five `@rule` patterns of `generate_nonlinlap_rules` against the additive
`terms`, keeping the last match per term (mirrors the scalar path's `Dict` semantics).
Grid-varying prefactors throw: the scalar path leaves them undiscretized, so no slice
form can reproduce them.
"""
function match_nonlinlap_terms(terms, s, depvars)
    ruleobjs = []
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule *(
                ~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d
            ) => _nllap_match(x, u, [~a..., ~b...], [~c..., ~d...], nothing)
        )
    end
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule $(Differential(x))(
                *(~~a, $(Differential(x))(u), ~~b)
            ) => _nllap_match(x, u, [~a..., ~b...], [], nothing)
        )
    end
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule (
                $(Differential(x))($(Differential(x))(u) / ~a)
            ) => _nllap_match_div(x, u, ~a, [], nothing)
        )
    end
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule *(
                ~~b, ($(Differential(x))($(Differential(x))(u) / ~a)), ~~c
            ) => _nllap_match_div(x, u, ~a, [~b..., ~c...], nothing)
        )
    end
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule /(
                *(~~b, ($(Differential(x))(*(~~a, $(Differential(x))(u), ~~d))), ~~c), ~e
            ) => _nllap_match(x, u, [~a..., ~d...], [~b..., ~c...], ~e)
        )
    end

    matches = NonlinlapMatch[]
    for t in terms
        m = nothing
        for r in ruleobjs
            v = r(t)
            v === nothing || (m = v)
        end
        m === nothing && continue
        if m.pre !== nothing
            pre = safe_unwrap(m.pre)
            if !isempty(get_depvars(pre, s.vars.depvar_ops)) ||
                    any(y -> subsmatch(pre, safe_unwrap(y) => nothing), s.x̄)
                throw(
                    ArrayDiscretizationFallback(
                        "grid-varying factor $(m.pre) multiplying a nonlinear laplacian, which the scalar path leaves undiscretized"
                    )
                )
            end
        end
        push!(matches, NonlinlapMatch(t, m.x, m.u, m.expr, m.pre, m.div))
    end
    return matches
end

"""
Derivative orders above one that the coefficient applies along `m.x`; order one is
always contributed by the inner derivative. Accepts `NonlinlapMatch` and
`SphericalMatch`.
"""
function nonlinlap_coeff_orders(m, depvars, derivweights)
    expr = safe_unwrap(m.expr)
    orders = Int[]
    for d in unique(vcat(1, derivweights.orders[m.x]))
        d == 1 && continue
        used = any(
            v -> subsmatch(expr, safe_unwrap((Differential(m.x)^d)(v)) => nothing),
            depvars
        )
        used && push!(orders, d)
    end
    return orders
end

# Interior-branch tap offsets of a half-offset operator (`Itap` in
# `get_half_offset_weights_and_stencil`).
half_offset_taps(D) = (1 - div(D.stencil_length, 2)):div(D.stencil_length, 2)

"""
Band bounds and tap extents `(lo, hi, mintap, maxtap)` a nonlinear laplacian imposes in
direction `x`: the interior branch conditions of `get_half_offset_weights_and_stencil`
for the outer operator (applied at `II - 1` on the clipped length `n - 1`) and, at each
of its half-offset taps, for the interpolator and the inner half-offset derivatives.
"""
function array_nonlinlap_constraints(x, n, derivweights, coeff_orders)
    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    bpc_o = D_outer.boundary_point_count
    # Outer interior branch: bpc_o < II - 1 <= (n-1) - bpc_o.
    lo = bpc_o + 2
    hi = n - bpc_o
    outer_taps = half_offset_taps(D_outer) .- 1
    mintap, maxtap = 0, 0
    inner_ops = vcat(
        [derivweights.interpmap[x], derivweights.halfoffsetmap[1][Differential(x)]],
        [derivweights.halfoffsetmap[1][Differential(x)^d] for d in coeff_orders]
    )
    for D in inner_ops
        bpc = D.boundary_point_count
        # Interior branch at each half-offset point: bpc < II + o <= n - bpc.
        lo = max(lo, bpc + 1 - first(outer_taps))
        hi = min(hi, n - bpc - last(outer_taps))
        taps = half_offset_taps(D)
        mintap = min(mintap, first(outer_taps) + first(taps))
        maxtap = max(maxtap, last(outer_taps) + last(taps))
    end
    return lo, hi, mintap, maxtap
end

"""
Slice form of the half-offset operator `D` applied to `v` at offset `o` from each core
point (interior branch of `get_half_offset_weights_and_stencil`). On nonuniform grids
the weights vary per point, indexed as `stencil_coefs[i + o - boundary_point_count]`.
"""
function array_half_offset_stencil(D, v, o, s, x, ranges, indexmap, periodic)
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]
    taps = half_offset_taps(D)
    slices = [
        array_slice(
                v, s, ranges, indexmap; shiftx = x, offset = o + q, periodic = periodic
            ) for q in taps
    ]
    weights = if D.dx isa Number
        collect(D.stencil_coefs)
    else
        [
            array_weight_vals(
                    i -> D.stencil_coefs[i + o - D.boundary_point_count], k, rng, j, N
                ) for k in eachindex(taps)
        ]
    end
    return array_stencil(weights, slices)
end

"""
Numeric grid values of `x` interpolated to the half-offset point at offset `o`, shaped
to broadcast along the dimension of `x` (slice form of `map_ivs_to_interpolated`, with
`_wrapperiodic`-style wrapping).
"""
function array_interp_grid_vals(interp, o, s, x, ranges, indexmap, periodic, N)
    j = indexmap[x]
    rng = ranges[j]
    grid = s.grid[x]
    n = length(s, x)
    taps = half_offset_taps(interp)
    wrap(i) = if haskey(periodic, x)
        i <= 1 ? i + n - 1 : (i > n ? i - n + 1 : i)
    else
        i
    end
    getweights(i) = interp.dx isa Number ? interp.stencil_coefs :
        interp.stencil_coefs[i + o - interp.boundary_point_count]
    vals = [dot(getweights(i), [grid[wrap(i + o + q)] for q in taps]) for i in rng]
    N == 1 && return vals
    return reshape(vals, ntuple(k -> k == j ? length(vals) : 1, N))
end

"""
Slice form of `cartesian_nonlinear_laplacian` for a matched `Dx(expr * Dx(u))`: at each
half-offset point of the outer stencil, `expr * Dx(u)` is rebuilt over shifted slices
(dependent variables interpolated, same-`x` derivatives via the half-offset operators,
`x` interpolated numerically), then combined with the outer weights. Unhandled patterns
in the coefficient surface as `ArrayDiscretizationFallback` from `arrayify`.
"""
function array_nonlinear_laplacian(
        m::NonlinlapMatch, s, depvars, derivweights, ranges, indexmap, periodic
    )
    x, u = m.x, m.u
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]

    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    interp = derivweights.interpmap[x]
    # Outer derivative is applied at II - 1 (`outerstencil` in the scalar path).
    outer_taps = half_offset_taps(D_outer) .- 1

    # Only build rules for orders the coefficient uses: slices are built eagerly and the
    # band only guarantees in-range taps for those operators.
    orders = vcat(1, nonlinlap_coeff_orders(m, depvars, derivweights))
    interpolated = map(outer_taps) do o
        rules = Pair[]
        for v in depvars
            for order in orders
                D = derivweights.halfoffsetmap[1][Differential(x)^order]
                push!(
                    rules,
                    safe_unwrap((Differential(x)^order)(v)) => array_half_offset_stencil(
                        D, v, o, s, x, ranges, indexmap, periodic
                    )
                )
            end
        end
        for v in depvars
            push!(
                rules,
                safe_unwrap(v) => array_half_offset_stencil(
                    interp, v, o, s, x, ranges, indexmap, periodic
                )
            )
        end
        push!(
            rules,
            safe_unwrap(x) => array_interp_grid_vals(
                interp, o, s, x, ranges, indexmap, periodic, N
            )
        )
        for y in ivs(depvar(u, s), s)
            isequal(y, x) && continue
            push!(rules, safe_unwrap(y) => array_grid_vals(y, s, ranges, indexmap, N))
        end
        ctx = ArrayifyContext(rules, s.time)
        return arrayify(m.expr * Differential(x)(u), ctx)
    end

    outerweights = if D_outer.dx isa Number
        collect(D_outer.stencil_coefs)
    else
        # -1: the outer derivative sits at II - 1.
        [
            array_weight_vals(
                    i -> D_outer.stencil_coefs[i - 1 - D_outer.boundary_point_count],
                    k, rng, j, N
                ) for k in eachindex(outer_taps)
        ]
    end
    return array_stencil(outerweights, interpolated)
end

"""
Rules binding each matched term to its slice form: the laplacian from
`array_nonlinear_laplacian`, grid-constant prefactors broadcast on, divisors
discretized with the base rules (mirrors `replacevals`).
"""
function array_nonlinlap_rules(
        matches, s, depvars, derivweights, ranges, indexmap, baserules, periodic
    )
    basectx = ArrayifyContext(baserules, s.time)
    rules = Pair[]
    for m in matches
        val = array_nonlinear_laplacian(
            m, s, depvars, derivweights, ranges, indexmap, periodic
        )
        m.pre === nothing || (val = broadcast(*, arrayify(m.pre, basectx), val))
        m.div === nothing || (val = broadcast(/, val, arrayify(m.div, basectx)))
        push!(rules, safe_unwrap(m.term) => val)
    end
    return rules
end

####
# Spherical laplacian in slice form
####

"""
A matched spherical laplacian `Dx(x^2 * expr * Dx(u)) / x^2`: `term` is the matched
additive term, `expr` the inner coefficient besides `x^2` and `pre` any prefactor
product (`nothing` when absent).
"""
struct SphericalMatch
    term::Any
    x::Any
    u::Any
    expr::Any
    pre::Any
end

"""
Match the three `@rule` patterns of `generate_spherical_diffusion_rules` against the
additive `terms`, keeping the last match per term. Terms carrying a nonlinear laplacian
match are skipped: the scalar path keys both rulesets by the same term and the nonlinear
laplacian entry, added later, wins its rules `Dict`. Grid-varying prefactors throw for
the same reason as in `match_nonlinlap_terms`.
"""
function match_spherical_terms(terms, s, depvars, nllap_matches)
    ruleobjs = []
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule *(
                ~~a, 1 / (x^2),
                ($(Differential(x))(*(~~c, (x^2), ~~d, $(Differential(x))(u), ~~e))),
                ~~b
            ) => SphericalMatch(
                nothing, x, u, *(~c..., ~d..., ~e..., Num(1)),
                _nllap_pre([~a..., ~b...])
            )
        )
    end
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule /(
                *(
                    ~~a,
                    ($(Differential(x))(*(~~c, (x^2), ~~d, $(Differential(x))(u), ~~e))),
                    ~~b
                ),
                (x^2)
            ) => SphericalMatch(
                nothing, x, u, *(~c..., ~d..., ~e..., Num(1)),
                _nllap_pre([~a..., ~b...])
            )
        )
    end
    for u in depvars, x in ivs(depvar(u, s), s)
        push!(
            ruleobjs,
            @rule /(
                ($(Differential(x))(*(~~c, (x^2), ~~d, $(Differential(x))(u), ~~e))),
                (x^2)
            ) => SphericalMatch(nothing, x, u, *(~c..., ~d..., ~e..., Num(1)), nothing)
        )
    end

    claimed = [safe_unwrap(m.term) for m in nllap_matches]
    matches = SphericalMatch[]
    for t in terms
        any(c -> isequal(safe_unwrap(t), c), claimed) && continue
        m = nothing
        for r in ruleobjs
            v = r(t)
            v === nothing || (m = v)
        end
        m === nothing && continue
        if m.pre !== nothing
            pre = safe_unwrap(m.pre)
            if !isempty(get_depvars(pre, s.vars.depvar_ops)) ||
                    any(y -> subsmatch(pre, safe_unwrap(y) => nothing), s.x̄)
                throw(
                    ArrayDiscretizationFallback(
                        "grid-varying factor $(m.pre) multiplying a spherical laplacian, which the scalar path leaves undiscretized"
                    )
                )
            end
        end
        push!(matches, SphericalMatch(t, m.x, m.u, m.expr, m.pre))
    end
    return matches
end

"""
Band bounds and tap extents `(lo, hi, mintap, maxtap)` a spherical laplacian imposes in
direction `x`: those of the nonlinear laplacian it contains, plus the interior branch of
the centered first derivative the scheme adds.
"""
function array_spherical_constraints(x, n, derivweights, coeff_orders)
    lo, hi, mintap, maxtap = array_nonlinlap_constraints(x, n, derivweights, coeff_orders)
    D_1 = derivweights.map[Differential(x)]
    bpc = D_1.boundary_point_count
    lo = max(lo, bpc + 1)
    hi = min(hi, n - bpc)
    taps = half_range(D_1.stencil_length)
    mintap = min(mintap, first(taps))
    maxtap = max(maxtap, last(taps))
    return lo, hi, mintap, maxtap
end

"""
Slice form of `spherical_diffusion` for a matched `Dx(x^2 * expr * Dx(u)) / x^2`, away
from x ≈ 0 (scheme 1 in appendix A of the paper referenced there): the coefficient at
the grid points times the centered first derivative divided by the grid values of `x`
plus the nonlinear laplacian of the inner coefficient.
"""
function array_spherical_diffusion(
        m::SphericalMatch, s, depvars, derivweights, ranges, indexmap, periodic, baserules
    )
    x, u = m.x, m.u
    N = length(ranges)
    D_1 = derivweights.map[Differential(x)]
    D1u = array_central_difference(D_1, s, u, x, 1, ranges, indexmap, periodic)
    xvals = array_grid_vals(x, s, ranges, indexmap, N)
    exprhere = arrayify(m.expr, ArrayifyContext(baserules, s.time))
    nll = array_nonlinear_laplacian(
        NonlinlapMatch(nothing, x, u, m.expr, nothing, nothing),
        s, depvars, derivweights, ranges, indexmap, periodic
    )
    return broadcast(*, exprhere, broadcast(+, broadcast(/, D1u, xvals), nll))
end

"""
Rules binding each matched spherical term to its slice form, grid-constant prefactors
broadcast on (mirrors the splicing in `generate_spherical_diffusion_rules`).
"""
function array_spherical_rules(
        matches, s, depvars, derivweights, ranges, indexmap, baserules, periodic
    )
    basectx = ArrayifyContext(baserules, s.time)
    rules = Pair[]
    for m in matches
        val = array_spherical_diffusion(
            m, s, depvars, derivweights, ranges, indexmap, periodic, baserules
        )
        m.pre === nothing || (val = broadcast(*, arrayify(m.pre, basectx), val))
        push!(rules, safe_unwrap(m.term) => val)
    end
    return rules
end

struct ArrayifyContext
    rules::Vector{<:Pair}
    time::Any
end

# symtype falls back to typeof for non-symbolic values, so this covers literal arrays,
# symbolic arrays and scalars of either kind.
is_array_valued(x) = SymbolicUtils.symtype(safe_unwrap(x)) <: AbstractArray

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
    elseif !(op isa Function)
        # Symbolic operators (`Integral`, ...) and symbolic callables are not `Function`s
        # and cannot be broadcast over slices; anything reaching here was not replaced by
        # a rule, so there is no slice form for it.
        throw(ArrayDiscretizationFallback("unhandled operation $op in $expr"))
    end
    newargs = [arrayify(a, ctx) for a in arguments(expr)]
    if any(is_array_valued, newargs)
        return broadcast(op, newargs...)
    else
        return op(newargs...)
    end
end

####
# Boundary equations in slice form
####

"""
    array_bc_eqs(s, boundary, interiormap, derivweights, bcmap)

Generate the equations for `boundary` as a single symbolic array equation over the face
it occupies, rather than one scalar equation per point on that face.

This works because the index along the boundary's own direction is fixed across the
face, so every point on it selects the same stencil weights and tap offsets — the same
translation invariance the interior exploits, applied one dimension down. Without this,
boundary equations stay pointwise and dominate the equation count in 2D and 3D, where
they scale with the surface (`O(n)` and `O(n^2)`) while the interior collapses to one.

Throws `ArrayDiscretizationFallback` for boundaries with no slice representation, in
which case the caller emits the pointwise form.
"""
function array_bc_eqs(s, boundary, interiormap, derivweights, bcmap)
    boundary isa AbstractTruncatingBoundary ||
        throw(ArrayDiscretizationFallback("non-truncating (interface) boundary"))

    u_, x_ = getvars(boundary)
    u = depvar(u_, s)
    args = ivs(u, s)
    length(args) == 0 && throw(ArrayDiscretizationFallback("no spatial arguments"))
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    haskey(indexmap, x_) ||
        throw(ArrayDiscretizationFallback("boundary variable $x_ not an argument of $u"))
    j = indexmap[x_]

    E = edge(s, boundary, interiormap)
    length(E) == 0 && throw(ArrayDiscretizationFallback("empty boundary edge"))
    lo = collect(Tuple(first(E)))
    hi = collect(Tuple(last(E)))
    # the face must be a contiguous box for a slice to describe it
    length(E) == prod(hi .- lo .+ 1) ||
        throw(ArrayDiscretizationFallback("boundary edge is not a contiguous box"))
    lo[j] == hi[j] ||
        throw(ArrayDiscretizationFallback("boundary edge spans its own direction"))
    ranges = Dict(i => lo[i]:hi[i] for i in eachindex(lo))
    N = length(args)
    # A single-point face (every 1D boundary) has nothing to collapse; a one-element
    # slice equation would just be a more convoluted spelling of the scalar one.
    prod(length(ranges[i]) for i in 1:N) == 1 &&
        throw(ArrayDiscretizationFallback("single-point boundary", true))
    # A staggered 1D boundary is always the single point above; multi-point staggered
    # faces would need the staggered stencil selection, which has no slice form here yet.
    get_grid_type(s) <: StaggeredGrid &&
        throw(ArrayDiscretizationFallback("staggered boundary face"))

    # Every dependent variable in the condition must be one this path can slice: either
    # the canonical variable, or a value on this same boundary.
    bcdepvars = get_depvars(boundary.eq.lhs, s.vars.depvar_ops) ∪
        get_depvars(boundary.eq.rhs, s.vars.depvar_ops)
    for v in bcdepvars
        vd = depvar(v, s)
        isequal(ivs(vd, s), args) ||
            throw(ArrayDiscretizationFallback("variable $v of differing dimensionality"))
        for (k, a) in enumerate(remove(arguments(v), s.time))
            unwrap_const(safe_unwrap(a)) isa Number || continue
            k == j || throw(
                ArrayDiscretizationFallback("boundary value of $v away from this boundary")
            )
        end
    end
    # An interface in this boundary's own direction wraps the stencil taps below onto
    # indices that are not a shift of the face; interfaces in the other directions do not
    # enter this equation at all.
    isempty(filter_interfaces(bcmap[operation(u)][x_])) ||
        throw(ArrayDiscretizationFallback("interface boundary condition in $x_"))

    II0 = first(E)
    ufunc(v, I, x) = s.discvars[v][I]

    # Derivatives in the boundary direction: take the weights and taps the scalar path
    # would use at a representative point on the face, then express the taps as shifted
    # slices. The branch that selects them depends only on the index along `x_`, which is
    # constant across the face, so this is exact.
    derivrules = Pair[]
    for d in derivweights.orders[x_]
        Dop = get(derivweights.map, Differential(x_)^d, nothing)
        Dop === nothing && continue
        ws, Itap = try
            central_difference_weights_and_stencil(
                Dop, II0, s, filter_interfaces(bcmap[operation(u)][x_]), (j, x_), u
            )
        catch e
            e isa InterruptException && rethrow(e)
            throw(ArrayDiscretizationFallback("could not build boundary stencil for order $d"))
        end
        offsets = [I[j] - II0[j] for I in Itap]
        all(I -> all(k -> k == j || I[k] == II0[k], 1:N), Itap) ||
            throw(ArrayDiscretizationFallback("boundary stencil is not axis aligned"))
        slices = [
            array_slice(u, s, ranges, indexmap; shiftx = x_, offset = o) for o in offsets
        ]
        expr = array_stencil(collect(ws), slices)
        for v in bcdepvars
            isequal(depvar(v, s), u) || continue
            push!(derivrules, safe_unwrap((Differential(x_)^d)(v)) => expr)
        end
        push!(derivrules, safe_unwrap((Differential(x_)^d)(u)) => expr)
    end

    # Dependent variables (both the canonical form and the value on this boundary) map to
    # the slice over the face.
    varrules = Pair[]
    for v in bcdepvars
        vd = depvar(v, s)
        push!(varrules, safe_unwrap(v) => array_slice(vd, s, ranges, indexmap))
    end
    push!(varrules, safe_unwrap(u) => array_slice(u, s, ranges, indexmap))

    # Independent variables: the boundary's own variable takes its endpoint value (a
    # scalar), the others vary along the face. This mirrors `axiesvals`.
    gridrules = Pair[]
    for x in args
        if isequal(x, x_)
            val = lo[j] == 1 ? first(s.axies[x]) : last(s.axies[x])
            push!(gridrules, safe_unwrap(x) => val)
        else
            push!(gridrules, safe_unwrap(x) => array_grid_vals(x, s, ranges, indexmap, N))
        end
    end

    ctx = ArrayifyContext(vcat(derivrules, varrules, gridrules), s.time)
    lhs = arrayify(boundary.eq.lhs, ctx)
    rhs = arrayify(boundary.eq.rhs, ctx)
    shape = Tuple(length(ranges[i]) for i in 1:N)
    if is_array_valued(lhs) && !is_array_valued(rhs)
        rhs = fill(Symbolics.unwrap(rhs), shape)
    elseif !is_array_valued(lhs) && is_array_valued(rhs)
        lhs = fill(Symbolics.unwrap(lhs), shape)
    elseif !is_array_valued(lhs) && !is_array_valued(rhs)
        throw(ArrayDiscretizationFallback("boundary condition has no discretizable terms"))
    end
    return [lhs ~ rhs]
end

"""
    array_bc_eqs(s, boundary::InterfaceBoundary, interiormap, derivweights, bcmap)

Equate the two faces an interface (periodic) boundary joins as a single array equation,
the slice form of the `disc1[II] ~ disc2[II + Ioffset]` the scalar path emits per point.

As in the scalar path only the lower boundary of the pair carries the equations; the upper
one repeats the same relation and contributes none.
"""
function array_bc_eqs(s, boundary::InterfaceBoundary, interiormap, derivweights, bcmap)
    isupper(boundary) && return Equation[]

    u = depvar(boundary.u, s)
    u2 = depvar(boundary.u2, s)
    args = ivs(u, s)
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    haskey(indexmap, boundary.x) ||
        throw(ArrayDiscretizationFallback("boundary variable $(boundary.x) not an argument of $u"))
    j = indexmap[boundary.x]
    N = length(args)

    E = edge(s, boundary, interiormap)
    length(E) == 0 && throw(ArrayDiscretizationFallback("empty boundary edge"))
    lo = collect(Tuple(first(E)))
    hi = collect(Tuple(last(E)))
    length(E) == prod(hi .- lo .+ 1) ||
        throw(ArrayDiscretizationFallback("boundary edge is not a contiguous box"))
    lo[j] == hi[j] ||
        throw(ArrayDiscretizationFallback("boundary edge spans its own direction"))
    ranges = Dict(i => lo[i]:hi[i] for i in eachindex(lo))
    length(E) == 1 &&
        throw(ArrayDiscretizationFallback("single-point interface boundary", true))
    # 1D staggered interfaces are the single point above; multi-point staggered faces
    # are untested on the scalar path, so decline rather than guess.
    get_grid_type(s) <: StaggeredGrid &&
        throw(ArrayDiscretizationFallback("staggered interface face"))

    arr2 = array_variable(u2, s)
    disc2 = s.discvars[u2]
    ndims(disc2) == N ||
        throw(ArrayDiscretizationFallback("interface joins variables of differing dimensionality"))
    # the same index shift `generate_bc_eqs!` applies pointwise
    shift = length(s, boundary.x2) - 1
    rs2 = ntuple(i -> i == j ? (ranges[i] .+ shift) : ranges[i], N)
    all(i -> checkindex(Bool, axes(disc2, i), rs2[i]), 1:N) ||
        throw(ArrayDiscretizationFallback("interface slice falls outside $(u2)"))

    return [array_slice(u, s, ranges, indexmap) ~ arr2[rs2...]]
end

"""
    array_corner_eqs(s, interiormap, u, N)

Equations for the points that lie outside the interior in two or more dimensions — the
corners in 2D, and the edges as well as the corners in 3D — as one array equation per
contiguous box instead of one scalar equation per point.

`generate_corner_eqs!` builds this region with `setdiff`, which yields a bag of indices
and loses the structure. The region is in fact a union of boxes: along each dimension a
point lies in the lower band, the interior band, or the upper band, and this region is
exactly the combinations with at least two non-interior bands. Enumerating those gives
`3^N - 2N - 1` boxes — none in 1D, the 4 corners in 2D, and the 12 edges plus 8 corners
in 3D — a count that does not depend on the grid resolution.

Without this the 3D edges stay pointwise and cost `12n - 10` equations, which is what
keeps 3D at `O(n)` once the faces are sliced.
"""
function array_corner_eqs(s, interiormap, u, N)
    N >= 2 || throw(ArrayDiscretizationFallback("no corner region below 2 dimensions", true))
    interior = interiormap.I[interiormap.pde[u]]
    length(interior) == 0 && throw(ArrayDiscretizationFallback("empty interior"))
    arr = array_variable(u, s)
    lo = Tuple(first(interior))
    hi = Tuple(last(interior))
    dims = size(s.discvars[u])

    # the three bands per dimension: below the interior, the interior, above it
    bands = map(1:N) do j
        (1:(lo[j] - 1), lo[j]:hi[j], (hi[j] + 1):dims[j])
    end

    eqs = Equation[]
    for combo in Iterators.product(ntuple(_ -> 1:3, N)...)
        count(!=(2), combo) >= 2 || continue          # needs >= 2 non-interior bands
        ranges = ntuple(j -> bands[j][combo[j]], N)
        any(isempty, ranges) && continue
        shape = map(length, ranges)
        if prod(shape) == 1
            # a single point is more clearly written as the scalar equation it is
            push!(eqs, s.discvars[u][CartesianIndex(map(first, ranges))] ~ 0)
        else
            push!(eqs, arr[ranges...] ~ zeros(shape))
        end
    end
    return eqs
end
