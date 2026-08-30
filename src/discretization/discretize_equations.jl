# Equation discretization using array slices.
#
# Instead of one scalar equation per interior grid point, the interior of each PDE is
# emitted as a single symbolic array equation over slices of the underlying array
# variables, e.g. for the 1D heat equation with second order approximation:
#
#   D(u[2:n-1]) ~ (u[1:n-2] .- 2u[2:n-1] .+ u[3:n]) ./ dx^2
#
# Boundary, extrapolation and corner equations reuse the pointwise machinery, as do any
# interior points close enough to a boundary that their stencil differs from the
# translation-invariant interior stencil (the "frame").
#
# Periodic and two-domain interface directions are translation invariant over the
# interior at a wrapping end — the pointwise path never selects a boundary stencil
# there, it wraps the taps across the seam (onto the same array, or another
# variable's array) — so the interior is emitted as one array equation per box of
# the decomposition described in `array_bands`, a count that does not depend on
# the grid resolution.
#
# Interior boundary values (e.g. `u(t, 1)`) map to the matching array element or face
# slice on every array box, including size-1 wrap boxes and frame points (the pointwise
# `boundaryvalfuncs` skip interface faces and free-standing corners; this path does not).
# On an edge-aligned grid the same references interpolate to the domain edge via
# `interpmap` (the pointwise `half_offset_centered_difference`). Pinned-direction
# derivatives of those values (`Dx(u(t, 1))`) use the same `depvarderivbcmaps` stencil
# on a center-aligned grid (one-sided at a truncating edge, wrapped if periodic).
# Time-literal references such as `u(0, x)` evaluate that variable's initial-condition
# formula on the matching slice. Derivatives of time-literals, non-IC times,
# edge-aligned boundary-value derivatives, and staggered-grid boundary values fall back.
#
# Nonlinear laplacians `Dx(a(u) * Dx(u))` are emitted in slice form too; see
# `array_nonlinear_laplacian`. Spherical laplacians `r^-2 Dr(r^2 Dr(u))` build on the
# same half-offset machinery; see `array_spherical_diffusion`.
#
# Functional advection schemes are traced once per direction, taps
# replaced by shifted slices; on nonuniform grids the coordinate arithmetic moves into
# numeric per-point coefficients (`array_scheme_split`). See `array_function_scheme_trace`
# for what falls back.
#
# Staggered grids collapse the same way: each variable's alignment fixes its two stencil
# taps across the interior, so the interior is one array equation per PDE; the (1D,
# single-point) boundaries stay pointwise.
#
# Mixed derivatives `(Differential(x)^m * Differential(y)^n)(u)` are the tensor product
# of the two centered stencils of those orders; see `array_mixed_difference`. The first-
# order case `Dx(Dy(u))` is `m = n = 1`. Three-or-more spatial directions, mixed
# derivatives of boundary values, and mixed derivatives on staggered grids still fall back.
#
# Integrals are reductions, not stencils: a cumulative `Integral(xmin, x)(u)` is a
# trapezoidal running sum along that axis (`axis_cumsum_pad` of the same increments
# `_euler_integral` walks); a whole-domain integral is a weighted `sum` and is emitted
# only when the integration axis is absent from the equation variable (the pointwise
# `indexmap` / `bvar` filter). Time-only dependents broadcast as scalars.
#
# Lower-dimensional fields (`v(t, x)` or `S(t)` in a `u(t, x, y)` equation) slice
# on their own axes and broadcast against the core. Non-prefix subsets are
# permuted and reshaped with singleton dimensions, matching `Idx`. A whole-domain
# integrand with extra axes is a reduction (`array_integral_rules`), not a slice.
#
# Unsupported patterns fall back to pointwise equations: callbacks,
# a higher-dimensional field whose extra axes are free (not a boundary
# value and not an integral rank drop), derivatives of time-literals,
# derivatives of boundary values on an edge-aligned grid, staggered-grid
# boundary values, tangential or mixed derivatives of boundary values,
# stationary systems, linear operators on a nonuniform two-domain interface.

struct ArrayFormFallback <: Exception
    msg::String
end

function discretize_equation_at_point(
        II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap, boundaryvalfuncs;
        extra_rules = []
    )
    boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs, init = [])
    # extra_rules last: Dict last-key-wins, and before expand_derivatives.
    rules = vcat(
        generate_finite_difference_rules(
            II, s, depvars, pde, derivweights, bcmap, indexmap
        ),
        boundaryrules,
        valmaps(s, eqvar, depvars, II, indexmap),
        extra_rules
    )
    try
        rdict = Dict(rules)
        return expand_derivatives(pde_substitute(pde.lhs, rdict)) ~
            pde_substitute(pde.rhs, rdict)
    catch e
        println("A scheme has been incorrectly applied to the following equation: $pde.\n")
        println("The following rules were constructed at index $II:")
        display(rules)
        rethrow(e)
    end
end

function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
        discretization::MOLFiniteDifference
    )
    # Boundary handling is shared with the pointwise fallback.
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
            reason = e isa ArrayFormFallback ? e.msg : sprint(showerror, e)
            @debug "Array form falling back to pointwise boundary equations for $(boundary.eq): $reason"
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
        reason = e isa ArrayFormFallback ? e.msg : sprint(showerror, e)
        @debug "Array form falling back to pointwise corner equations: $reason"
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
            # pointwise path can discretize into an error. Genuine problems with the equation
            # itself resurface below, where the pointwise path raises them directly.
            reason = e isa ArrayFormFallback ? e.msg :
                sprint(showerror, e)
            @debug "Array form falling back to pointwise discretization for $pde: $reason"
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
translation-invariant interior stencil. Throws `ArrayFormFallback` when `pde`
contains a pattern that cannot be represented this way.
"""
function discretize_equation_array_form(
        pde, interior, s, depvars, derivweights, bcmap,
        eqvar, indexmap, boundaryvalfuncs
    )
    isstag = get_grid_type(s) <: StaggeredGrid
    # Stationary: PDEBase emits `0 ~ residual`; Symbolics rejects scalar ~ array.
    s.time === nothing && throw(
        ArrayFormFallback(
            "stationary (no time) systems have no array form in NonlinearSystem construction"
        )
    )
    # The staggered path never consults the advection scheme (see the staggered
    # `generate_finite_difference_rules`), so the requirement only applies otherwise.
    isstag || derivweights.advection_scheme isa Union{UpwindScheme, FunctionalScheme} ||
        throw(
        ArrayFormFallback(
            "unsupported advection scheme $(derivweights.advection_scheme)"
        )
    )

    args = ivs(eqvar, s)
    array_validate_depvar_axes(pde, s, args)
    slicevars = array_sliceable_depvars(
        s, array_unique_depvars(array_pde_occurrences(pde, s), s), args
    )
    periodic = array_wrap_dims(s, slicevars, args, bcmap)

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
    mixedterms = isstag ? [] : array_mixed_terms(pde, slicevars, args, pdeorders, s)
    mixedorders = mixed_orders_by_direction(mixedterms)
    bands, clean = array_bands(
        interior, s, args, pdeorders, derivweights, indexmap, periodic, nllap_orders,
        sph_orders, mixedorders
    )
    any(isempty, bands) && throw(ArrayFormFallback("empty core region"))
    N = length(args)
    core = CartesianIndices(
        ntuple(j -> first(first(bands[j])):last(last(bands[j])), N)
    )

    # Probe the special-case rulesets at a representative core point, one whose stencils
    # do not wrap. Several of these generators return candidate rules unconditionally; the
    # pointwise path only applies a special scheme when a rule key occurs in the equation, so
    # fall back exactly when one does. Any firing rule means a scheme with no slice
    # representation here yet. Integrals have a slice form (`array_integral_rules`);
    # callbacks still do not. The staggered pointwise path applies none of these schemes,
    # so there is nothing to probe there; remaining unsupported patterns surface in
    # `arrayify` instead.
    if !isstag
        II0 = CartesianIndex(
            ntuple(j -> clean[j] === nothing ? first(core)[j] : first(bands[j][clean[j]]), N)
        )
        special_rules = vec(
            generate_cb_rules(II0, s, depvars, derivweights, bcmap, indexmap, terms)
        )
        for r in special_rules
            (subsmatch(pde.lhs, r) || subsmatch(pde.rhs, r)) &&
                throw(ArrayFormFallback("unsupported pattern $(r.first)"))
        end
    end
    array_validate_boundary_values(pde, s, derivweights)

    core_eqs = map(Iterators.product(map(eachindex, bands)...)) do combo
        rs = ntuple(j -> bands[j][combo[j]], N)
        ranges = Dict(j => rs[j] for j in 1:N)
        if prod(map(length, rs)) == 1
            # a single point is more clearly written as the scalar equation it is;
            # still apply array bval rules (periodic faces have empty boundaryvalfuncs)
            eq = discretize_equation_at_point(
                CartesianIndex(map(first, rs)), s, depvars, pde, derivweights,
                bcmap, eqvar, indexmap, boundaryvalfuncs;
                extra_rules = array_point_boundary_derivative_rules(
                    pde, s, ranges, indexmap, derivweights, periodic, bcmap
                )
            )
            return array_substitute_boundary_values(
                eq, pde, s, ranges, indexmap, derivweights, periodic, bcmap
            )
        end
        return array_core_equation(
            pde, ranges, s, slicevars, derivweights,
            args, pdeorders, indexmap, terms, periodic, nllap_matches, sph_matches,
            mixedterms, bcmap, depvars
        )
    end

    frame = setdiff(vec(collect(interior)), vec(collect(core)))
    frame_eqs = map(frame) do II
        ranges = Dict(j => II[j]:II[j] for j in 1:N)
        eq = discretize_equation_at_point(
            II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap, boundaryvalfuncs;
            extra_rules = array_point_boundary_derivative_rules(
                pde, s, ranges, indexmap, derivweights, periodic, bcmap
            )
        )
        return array_substitute_boundary_values(
            eq, pde, s, ranges, indexmap, derivweights, periodic, bcmap
        )
    end
    return vcat(vec(core_eqs), frame_eqs)
end

"""
The array equation for one box of the interior, given as `ranges` (dimension => index
range). `depvars` are the sliceable fields; `allvars` includes rank-dropping integrands.
`bcmap` builds pinned-direction derivatives of interior boundary values.
"""
function array_core_equation(
        pde, ranges, s, depvars, derivweights, args, pdeorders, indexmap, terms,
        periodic, nllap_matches, sph_matches, mixedterms, bcmap, allvars = depvars
    )
    N = length(args)
    # Deriv rules before bare bvals and time-literals, both before field rules.
    # Keys are disjoint (`Dx(u(t, 1))` ≠ `u(t, 1)` ≠ `u(0, x)` ≠ `u(t, x)`).
    bvalrules = vcat(
        array_boundary_value_derivative_rules(
            pde, s, ranges, indexmap, derivweights, periodic, bcmap
        ),
        array_boundary_value_rules(pde, s, ranges, indexmap, derivweights),
        array_time_literal_rules(pde, s, ranges, indexmap, derivweights)
    )
    varrules = Pair[]
    for u in depvars
        if isempty(ivs(u, s))
            push!(varrules, safe_unwrap(u) => array_scalar_discvar(u, s))
        else
            push!(varrules, safe_unwrap(u) => array_slice(u, s, ranges, indexmap))
        end
    end
    isstag = get_grid_type(s) <: StaggeredGrid
    # Integrands may carry axes that the equation core does not.
    intrules = array_integral_rules(
        s, allvars, ranges, indexmap; staggered = isstag
    )
    gridrules = [
        safe_unwrap(x) => array_grid_vals(x, s, ranges, indexmap, N) for x in args
    ]
    if isstag
        derivrules = array_staggered_rules(
            s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
        )
        mixedrules = Pair[]
        windrules = Pair[]
        nllaprules = Pair[]
        sphrules = Pair[]
        advrules = Pair[]
    else
        derivrules = array_cartesian_rules(
            s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
        )
        mixedrules = array_mixed_rules(
            mixedterms, s, derivweights, ranges, indexmap, periodic
        )
        # Include bvalrules so `u(t, 1)*Dx(u)`, `Dx(u(t, 1))*Dx(u)`, and
        # `u(0, x)*Dx(u)` substitute first.
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
            pde, s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
        )
    end
    # Family order is the reverse of the pointwise path's last-key-wins `Dict`.
    # Integrals sit with the other operators, before the bare-variable rules.
    ctx = ArrayifyContext(
        vcat(
            bvalrules, mixedrules, windrules, advrules, nllaprules, sphrules,
            intrules, derivrules, varrules, gridrules
        ),
        s.time
    )

    lhs = arrayify(pde.lhs, ctx)
    rhs = arrayify(pde.rhs, ctx)
    # `~` rejects array ~ scalar. Broadcast a 0D scalar onto the core slice,
    # not a lower-dimensional aligned field that happens to come first.
    if is_array_valued(lhs) && !is_array_valued(rhs)
        rhs = array_broadcast_onto(rhs, array_shape_donor(varrules, ranges))
    elseif !is_array_valued(lhs) && is_array_valued(rhs)
        lhs = array_broadcast_onto(lhs, array_shape_donor(varrules, ranges))
    elseif !is_array_valued(lhs) && !is_array_valued(rhs)
        throw(ArrayFormFallback("equation contains no discretizable terms"))
    end
    return lhs ~ rhs
end

"""
Wrap specification for one direction: source length `n` and, per end, where a tap
that leaves the grid is read from.

`lower` / `upper` are `nothing` (that end is a regular boundary; shrink the core),
`:self` (self-periodic: wrap onto the same array with the same length), or
`(; u, x, n)` naming the partner variable a two-domain interface joins.
"""
array_wrap_is_twodomain(spec) =
    spec.lower isa NamedTuple || spec.upper isa NamedTuple

array_wrap_dest_n(dest, n_src) = dest isa NamedTuple ? dest.n : n_src

function array_mentions_iv(expr, x)
    expr === nothing && return false
    xe = safe_unwrap(x)
    return any(v -> isequal(safe_unwrap(v), xe), Symbolics.get_variables(expr))
end

"""
The wrap destination of interface boundary `b` on `u` along `x`: `:self` when the
join is the same variable at the other end of the same independent variable, otherwise
the partner's discrete variable, independent variable and grid length.

Throws when the join is the same end of both domains, or when the partner's array
layout is not the same `CartesianIndex` the pointwise `wrapinterface` writes into.
"""
function array_wrap_dest(s, u, x, b::InterfaceBoundary{B, B}) where {B}
    throw(
        ArrayFormFallback(
            "interface $(b.eq) joins two variables at the same end of the domain"
        )
    )
end

function array_wrap_dest(s, u, x, b::InterfaceBoundary)
    u = depvar(u, s)
    u2 = depvar(b.u2, s)
    x2 = b.x2
    isequal(depvar(b.u, s), u) && isequal(b.x, x) || throw(
        ArrayFormFallback("interface $(b.eq) is not a join of $u along $x")
    )
    src_j = x2i(s, u, x)
    src_j === nothing && throw(
        ArrayFormFallback("interface $(b.eq) is not a join of $u along $x")
    )
    interface_layout_compatible(s, b, src_j) || throw(
        ArrayFormFallback(
            "interface $(b.eq) joins variables with incompatible layout"
        )
    )
    isequal(u2, u) && isequal(x2, x) && return :self
    return (u = u2, x = x2, n = length(s, x2))
end

"""
The directions in which every dependent variable wraps — self-periodic or two-domain
interface — mapped to a wrap specification.

A self-periodic direction is one whose interface boundaries join a variable to itself at
the other end of the same independent variable, which is what `u(t, 0) ~ u(t, 1)` parses
to. A two-domain interface joins different variables (typically at one end only);
`haslowerupper` still reports that end as an interface, so the interior stencil applies
there with taps wrapped onto the partner array by `bwrap`, which `wrap_tap_range`
reproduces on slices.

A nonuniform wrapping direction is admitted: operators whose seam form the pointwise path
cannot build (linear stencils, half-offset operators) throw at their own sites instead.
"""
function array_wrap_dims(s, depvars, args, bcmap)
    periodic = Dict()
    for x in args
        candidates = filter(u -> array_iv_in(x, ivs(u, s)), depvars)
        isempty(candidates) && continue
        withiface = filter(u -> array_has_interface(bcmap, u, x), candidates)
        isempty(withiface) && continue
        length(withiface) == length(candidates) ||
            throw(ArrayFormFallback("interface boundaries on only some variables in $x"))
        specs = map(withiface) do u
            bs = filter_interfaces(bcmap[operation(u)][x])
            lower = nothing
            upper = nothing
            for b in bs
                dest = array_wrap_dest(s, u, x, b)
                if isupper(b)
                    upper === nothing || throw(
                        ArrayFormFallback("multiple upper interfaces for $u in $x")
                    )
                    upper = dest
                else
                    lower === nothing || throw(
                        ArrayFormFallback("multiple lower interfaces for $u in $x")
                    )
                    lower = dest
                end
            end
            if lower === :self || upper === :self
                (lower === :self && upper === :self) || throw(
                    ArrayFormFallback("interface boundary at one end of $x only")
                )
            end
            (lower === nothing && upper === nothing) && throw(
                ArrayFormFallback("interface boundaries in $x did not produce a wrap")
            )
            return (n = length(s, x), lower = lower, upper = upper)
        end
        twodomain = any(array_wrap_is_twodomain, specs)
        if twodomain
            length(specs) == 1 || throw(
                ArrayFormFallback(
                    "two-domain interface on more than one variable in $x"
                )
            )
            periodic[x] = only(specs)
        else
            periodic[x] = (n = length(s, x), lower = :self, upper = :self)
        end
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

A mixed derivative reaches along `x` with the *centered* stencil of the mixed order in
that direction, rather than the winding one `pdeorders` would select for an odd order, so
those orders take their centered taps too. For order 1 this is the first-order centered
stencil, which at approximation orders 4/6 is wider than the winding stencil.
"""
function array_tap_extents(x, pdeorders, derivweights, ::Type{G}, mixedorders) where {G}
    mintap = 0
    maxtap = 0
    if haskey(mixedorders, x)
        for m in mixedorders[x]
            taps = half_range(derivweights.map[Differential(x)^m].stencil_length)
            mintap = min(mintap, first(taps))
            maxtap = max(maxtap, last(taps))
        end
    end
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

function array_tap_extents(
        x, pdeorders, derivweights, ::Type{G}, mixedorders
    ) where {G <: StaggeredGrid}
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

A wrapping end — self-periodic or two-domain — has no such boundary branch: the interior
stencil applies through that end, but for points within a stencil of the seam some taps
wrap and the wrapped tap is not contiguous with the rest. Splitting those points off as
one range each keeps every tap of every box a single contiguous slice, at the cost of a
handful of extra equations — as many as the stencil is wide, so the count still does not
depend on the grid resolution. A direction may wrap at one end only; the other end still
shrinks as a regular boundary.

`nllap_orders` maps each direction carrying a nonlinear laplacian to the coefficient's
derivative orders above one; those directions additionally take the half-offset branch
conditions and tap extents of `array_nonlinlap_constraints`. `sph_orders` does the same
for spherical laplacians via `array_spherical_constraints`, and additionally keeps any
r ≈ 0 points out of the core (the pointwise path treats them with a separate branch).
`mixedorders` maps each direction a mixed derivative reaches along to the centered
orders used there. Those use the centered stencil of that order, which for an odd
mixed order is wider than the winding stencil the same entry of `pdeorders` would
select — at first order and `approx_order` 4/6 this is the original mixed first-order case.
"""
function array_bands(
        interior, s, args, pdeorders, derivweights, indexmap, periodic,
        nllap_orders, sph_orders, mixedorders
    )
    N = length(args)
    bands = [UnitRange{Int}[] for _ in 1:N]
    clean = Vector{Union{Nothing, Int}}(nothing, N)
    for x in args
        j = indexmap[x]
        lo = first(interior)[j]
        hi = last(interior)[j]
        n = length(s, x)
        mintap, maxtap = array_tap_extents(
            x, pdeorders, derivweights, get_grid_type(s), mixedorders
        )
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
                ArrayFormFallback("spherical laplacian in a periodic direction")
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
        wlower = haskey(periodic, x) ? periodic[x].lower : nothing
        wupper = haskey(periodic, x) ? periodic[x].upper : nothing
        # Taps must stay in range at a non-wrapping end, and no point may take a
        # boundary branch there: the centered one at II <= boundary_point_count or
        # II > n - boundary_point_count, the positive winding at II <= offside, the
        # negative one at II > n - boundary_point_count. The staggered branch
        # conditions use the centered operator's boundary_point_count for every order.
        # A wrapping end keeps the interior stencil and is split below.
        if wlower === nothing
            lo = max(lo, 1 - mintap)
        end
        if wupper === nothing
            hi = min(hi, n - maxtap)
        end
        for d in pdeorders[x]
            if get_grid_type(s) <: StaggeredGrid
                bpc = derivweights.map[Differential(x)^d].boundary_point_count
                wlower === nothing && (lo = max(lo, bpc + 1))
                wupper === nothing && (hi = min(hi, n - bpc))
            elseif iseven(d)
                bpc = derivweights.map[Differential(x)^d].boundary_point_count
                wlower === nothing && (lo = max(lo, bpc + 1))
                wupper === nothing && (hi = min(hi, n - bpc))
            elseif array_functional_advection(derivweights, d)
                # the branch conditions of `get_f_taps_coords`: points within
                # `length(F.lower)` of the start, or `length(F.upper)` of the end, take
                # a boundary function of the scheme instead of its interior one
                F = derivweights.advection_scheme
                wlower === nothing && (lo = max(lo, length(F.lower) + 1))
                wupper === nothing && (hi = min(hi, n - length(F.upper)))
            else
                wlower === nothing &&
                    (lo = max(lo, derivweights.windmap[2][Differential(x)^d].offside + 1))
                wupper === nothing && (
                    hi = min(
                        hi,
                        n - derivweights.windmap[1][Differential(x)^d].boundary_point_count
                    )
                )
            end
        end
        if haskey(mixedorders, x)
            for m in mixedorders[x]
                bpc = derivweights.map[Differential(x)^m].boundary_point_count
                wlower === nothing && (lo = max(lo, bpc + 1))
                wupper === nothing && (hi = min(hi, n - bpc))
            end
        end
        if nl !== nothing
            wlower === nothing && (lo = max(lo, nl[1]))
            wupper === nothing && (hi = min(hi, nl[2]))
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
                ArrayFormFallback(
                    "spherical laplacian with r ≈ 0 inside the core"
                )
            )
        end
        if wlower === nothing && wupper === nothing
            lo > hi && return bands, clean
            bands[j] = [lo:hi]
            clean[j] = 1
            continue
        end
        lo > hi && return bands, clean
        wraps(i) = (wlower !== nothing && i + mintap <= 1) ||
            (wupper !== nothing && i + maxtap > n)
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
The index range a tap slice takes across a wrapping seam, mirroring `_wrapinterface`:
indices at or below the first point come from the lower-end destination, indices past
the last point from the upper-end destination. A range that straddles the seam is not a
slice.

Returns `(dest, range)` where `dest` is `nothing` (same array, unshifted interior),
`:self` (same array, remapped across a periodic seam), or the partner named tuple of a
two-domain interface. `wrap_periodic_range` is the self-periodic special case.
"""
function wrap_tap_range(r, spec)
    lo, hi = first(r), last(r)
    n = spec.n
    # Entirely past the last point: upper wrap. `_wrapinterface` maps n+k → dest[k+1].
    if lo > n
        dest = spec.upper
        dest === nothing && throw(
            ArrayFormFallback("tap leaves a non-wrapping end")
        )
        return dest, (lo - n + 1):(hi - n + 1)
    end
    # Entirely at or below the first point. Index 1 wraps only when this end is an
    # interface (`_wrapperiodic`: I[j] <= 1); on a regular lower boundary it is a
    # valid source index (the Dirichlet/Neumann face).
    if hi <= 1
        dest = spec.lower
        if dest === nothing
            (lo >= 1 && hi <= n) && return nothing, r
            throw(ArrayFormFallback("tap leaves a non-wrapping end"))
        end
        n′ = array_wrap_dest_n(dest, n)
        return dest, (lo + n′ - 1):(hi + n′ - 1)
    end
    # Touches the source grid. A lower interface wraps index 1, so a range that
    # includes both 1 and 2+ is not a single slice.
    if lo >= 1 && hi <= n
        spec.lower !== nothing && lo <= 1 && throw(
            ArrayFormFallback("interface stencil tap straddles the seam")
        )
        return nothing, r
    end
    throw(ArrayFormFallback("interface stencil tap straddles the seam"))
end

function wrap_periodic_range(r, n::Integer)
    _, r′ = wrap_tap_range(r, (n = n, lower = :self, upper = :self))
    return r′
end

array_iv_in(x, xs) = any(y -> isequal(x, y), xs)

array_ivs_subset(uivs, args) = all(x -> array_iv_in(x, args), uivs)

function array_has_interface(bcmap, u, x)
    haskey(bcmap, operation(u)) || return false
    haskey(bcmap[operation(u)], x) || return false
    return !isempty(filter_interfaces(bcmap[operation(u)][x]))
end

"""
Dependent-variable occurrences on both sides of `pde`.
"""
array_pde_occurrences(pde, s) =
    get_depvars(pde.lhs, s.vars.depvar_ops) ∪
    get_depvars(pde.rhs, s.vars.depvar_ops)

"""
The canonical fields named by `occs`, uniqued by `isequal`.
"""
function array_unique_depvars(occs, s)
    uniques = []
    for u_ in occs
        u = depvar(u_, s)
        any(v -> isequal(u, v), uniques) || push!(uniques, u)
    end
    return uniques
end

"""
Dependent variables whose spatial IVs are a subset of `args` (including 0D).
Rank-dropping integrands are excluded; `array_integral_rules` handles them.
"""
array_sliceable_depvars(s, depvars, args) =
    filter(u -> array_ivs_subset(ivs(u, s), args), depvars)

"""
True when an occurrence of a higher-dimensional variable leaves an extra IV
free. A face reference such as `φ(t, x, 1.0)` is not free.
"""
function array_occurrence_has_free_extra_ivs(u_, u, s, args)
    uivs = ivs(u, s)
    extra = [y for y in uivs if !array_iv_in(y, args)]
    isempty(extra) && return false
    args_ = remove(arguments(u_), s.time)
    length(args_) == length(uivs) || return true
    for (y, a) in zip(uivs, args_)
        array_iv_in(y, args) && continue
        unwrap_const(safe_unwrap(a)) isa Number || return true
    end
    return false
end

"""
Equation-level check: every dependent-variable occurrence is a subset of the
equation axes (including 0D), an integral rank drop (`array_compatible_depvar`),
or a boundary value that fixes every extra IV.
"""
function array_validate_depvar_axes(pde, s, args)
    occs = array_pde_occurrences(pde, s)
    for u_ in occs
        u = depvar(u_, s)
        array_ivs_subset(ivs(u, s), args) && continue
        array_compatible_depvar(u_, args, pde, s) && continue
        array_occurrence_has_free_extra_ivs(u_, u, s, args) &&
            throw(ArrayFormFallback("variables of differing dimensionality"))
    end
    return nothing
end

function array_eq_axes(indexmap, N)
    ax = Vector{Any}(undef, N)
    for (x, j) in indexmap
        ax[j] = x
    end
    return ax
end

function array_is_prefix_aligned(uivs, eq_axes)
    length(uivs) > length(eq_axes) && return false
    return all(i -> isequal(uivs[i], eq_axes[i]), eachindex(uivs))
end

# Axes live in the type. `reshape(A, dims::Tuple)` reconstructs as a vector;
# `promote_symtype` / `promote_shape` keep the axes after `maketerm`.
struct AxisReshape{N, S} <: Function end

(op::AxisReshape{N, S})(A) where {N, S} = reshape(A, S)

function SymbolicUtils.promote_symtype(::AxisReshape{N}, ::Type) where {N}
    return Array{Real, N}
end

function SymbolicUtils.promote_shape(
        ::AxisReshape{N, S}, ::SymbolicUtils.ShapeT
    ) where {N, S}
    return SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, S))
end

"""
`reshape` as a symbolic array term. Promotion methods keep the axes.
"""
function array_reshape(sl, shape::NTuple{N, Int}) where {N}
    N == ndims(sl) && Tuple(size(sl)) == shape && return sl
    slu = safe_unwrap(sl)
    sh = SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, shape))
    t = SymbolicUtils.term(
        AxisReshape{N, shape}(), slu; type = Array{Real, N}, shape = sh
    )
    return Symbolics.wrap(t)
end

struct AxisPermutedims{N, P, S} <: Function end

(op::AxisPermutedims{N, P, S})(A) where {N, P, S} = permutedims(A, P)

function SymbolicUtils.promote_symtype(::AxisPermutedims{N}, ::Type) where {N}
    return Array{Real, N}
end

function SymbolicUtils.promote_shape(
        ::AxisPermutedims{N, P, S}, ::SymbolicUtils.ShapeT
    ) where {N, P, S}
    return SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, S))
end

"""
`permutedims` as a symbolic array term. Promotion methods keep the axes.
"""
function array_permutedims(sl, perm::NTuple{N, Int}) where {N}
    N <= 1 && return sl
    issorted(perm) && return sl
    slu = safe_unwrap(sl)
    sz = size(sl)
    newsz = ntuple(i -> Int(sz[perm[i]]), N)
    sh = SymbolicUtils.ShapeVecT(map(Base.UnitRange{Int} ∘ Base.OneTo, newsz))
    t = SymbolicUtils.term(
        AxisPermutedims{N, perm, newsz}(), slu; type = Array{Real, N}, shape = sh
    )
    return Symbolics.wrap(t)
end

"""
Align a slice of `u` (in `u`'s IV order) to the equation axes in `ranges`.
Prefix axes are left as-is; any other subset is permuted and reshaped with
singleton dimensions — the array form of `Idx`.
"""
function array_align_axes(sl, u, s, ranges, indexmap)
    u = depvar(u, s)
    uivs = ivs(u, s)
    N = length(ranges)
    isempty(uivs) && return sl
    eq_axes = array_eq_axes(indexmap, N)
    array_is_prefix_aligned(uivs, eq_axes) && return sl
    src = [indexmap[x] for x in uivs]
    aligned = array_permutedims(sl, Tuple(sortperm(src)))
    length(uivs) == N && return aligned
    target = ntuple(j -> array_iv_in(eq_axes[j], uivs) ? length(ranges[j]) : 1, N)
    return array_reshape(aligned, target)
end

"""
The underlying (unscalarized) array variable of which `s.discvars[u]` holds the elements.
"""
function array_variable(u, s)
    el = safe_unwrap(first(vec(s.discvars[u])))
    (iscall(el) && operation(el) === getindex) ||
        throw(ArrayFormFallback("discrete variable for $u is not an array variable"))
    arr = first(arguments(el))
    # For an array-valued dependent variable (`@variables u(..)[1:n]`) the discrete
    # variable is a nested getindex, so the immediate parent is a component rather than
    # the grid-shaped array this path can slice.
    T = SymbolicUtils.symtype(arr)
    (T <: AbstractArray && ndims(T) == length(ivs(u, s))) ||
        throw(ArrayFormFallback("discrete variable for $u is not a grid-shaped array"))
    return Symbolics.wrap(arr)
end

"""
A slice of the array variable for `u` over the core region, shifted by `offsets[j]` in each
dimension `j` it names and wrapped around the seam where that dimension wraps (see
`wrap_tap_range`). A two-domain wrap reads the partner array; a self-periodic wrap stays
on `u`. Dimensions absent from `offsets` are taken unshifted and unwrapped, which is
what the pointwise path does with the dimensions a stencil does not reach along.

Naming a dimension with offset `0` is not the same as omitting it: `_wrapperiodic` maps the
first index of a periodic dimension onto the last for every tap of a stencil that reaches
along it, the centre tap included. A destination wrap is applied once: the partner
array is not wrapped again, matching `wrapinterface` on a `RefCartesianIndex`.
"""
function array_shifted_slice(u, s, ranges, indexmap, offsets, periodic)
    u = depvar(u, s)
    uivs = ivs(u, s)
    isempty(uivs) && return array_scalar_discvar(u, s)
    all(y -> haskey(indexmap, y), uivs) ||
        throw(ArrayFormFallback("variable $u of differing dimensionality"))
    arr = array_variable(u, s)
    dest_u = nothing
    rs = map(uivs) do y
        j = indexmap[y]
        r = ranges[j]
        haskey(offsets, j) || return r
        r = r .+ offsets[j]
        (periodic !== nothing && haskey(periodic, y)) || return r
        dest, r′ = wrap_tap_range(r, periodic[y])
        if dest isa NamedTuple
            dest_u === nothing || isequal(dest_u, dest.u) || throw(
                ArrayFormFallback(
                    "mixed wrap onto two different variables in one stencil tap"
                )
            )
            dest_u = dest.u
        end
        return r′
    end
    if dest_u !== nothing
        arr = array_variable(dest_u, s)
        disc = s.discvars[dest_u]
        ndims(disc) == length(rs) || throw(
            ArrayFormFallback("interface joins variables of differing dimensionality")
        )
        all(i -> checkindex(Bool, axes(disc, i), rs[i]), eachindex(rs)) || throw(
            ArrayFormFallback("interface slice falls outside $(dest_u)")
        )
    end
    return array_align_axes(arr[rs...], u, s, ranges, indexmap)
end

"""
A slice of the array variable for `u` over the core region, optionally shifted by
`offset` in the dimension of `shiftx`, wrapped around the seam if that dimension is
periodic (see `wrap_periodic_range`). A 0D variable is the matching scalar.
The slice is aligned to the equation axes.
"""
function array_slice(u, s, ranges, indexmap; shiftx = nothing, offset = 0, periodic = nothing)
    u = depvar(u, s)
    isempty(ivs(u, s)) && return array_scalar_discvar(u, s)
    offsets = shiftx === nothing ? Dict{Int, Int}() : Dict(indexmap[shiftx] => offset)
    return array_shifted_slice(u, s, ranges, indexmap, offsets, periodic)
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
            ArrayFormFallback(
                "boundary value is not at a domain edge for $x = $xval"
            )
        )
    end
end

"""True when the argument of `u_` for `x` is numeric."""
function array_iv_is_pinned(u_, x, s)
    u = depvar(u_, s)
    args = ivs(u, s)
    args_ = remove(arguments(u_), s.time)
    length(args_) == length(args) || return false
    for (j, a) in enumerate(args_)
        isequal(args[j], x) || continue
        return unwrap_const(safe_unwrap(a)) isa Number
    end
    return false
end

"""True when `expr` is `(Differential(x)^d)(u_)` with `x` numeric on `u_`."""
function array_is_boundary_value_derivative(expr, s)
    expr = safe_unwrap(expr)
    iscall(expr) || return false
    op = operation(expr)
    op isa Differential || return false
    (s.time !== nothing && isequal(op.x, s.time)) && return false
    op.order isa Integer || return false
    arg = safe_unwrap(only(arguments(expr)))
    iscall(arg) || return false
    any(u -> isequal(operation(arg), u), s.vars.depvar_ops) || return false
    any(a -> unwrap_const(safe_unwrap(a)) isa Number, arguments(arg)) || return false
    return array_iv_is_pinned(arg, op.x, s)
end

function array_collect_boundary_value_derivatives!(out, expr, s)
    expr = safe_unwrap(expr)
    iscall(expr) || return out
    if array_is_boundary_value_derivative(expr, s)
        op = operation(expr)
        arg = safe_unwrap(only(arguments(expr)))
        any(t -> isequal(t[1], expr), out) ||
            push!(out, (expr, op.x, Int(op.order), arg))
        return out
    end
    for a in arguments(expr)
        array_collect_boundary_value_derivatives!(out, a, s)
    end
    return out
end

"""Handleable `(expr, x, d, u_)` records in `pde`, e.g. `Dx(u(t, 1))`."""
function array_boundary_value_derivative_terms(pde, s)
    found = []
    array_collect_boundary_value_derivatives!(found, pde.lhs, s)
    array_collect_boundary_value_derivatives!(found, pde.rhs, s)
    return found
end

"""
Numeric literal in the time slot of `u_`, or `nothing`.

Aligns `arguments(u_)` with `s.args[operation(u_)]` so `u(x, 0)` is recognized
when time is not first. A bare Number test would read `u(0, x)` as the spatial
edge `u(t, 0)` whenever `0` is the left end of `x`.
"""
function array_time_slot_literal(u_, s)
    s.time === nothing && return nothing
    canon = s.args[operation(u_)]
    occ = arguments(u_)
    length(canon) == length(occ) || return nothing
    t0 = nothing
    for (c, a) in zip(canon, occ)
        if isequal(c, s.time)
            aval = unwrap_const(safe_unwrap(a))
            aval isa Number || return nothing
            t0 = aval
        end
    end
    return t0
end

array_is_time_literal_term(u_, s) = array_time_slot_literal(u_, s) !== nothing

function array_numeric_arg_depvars(pde, s)
    terms = []
    for u in get_depvars(pde.lhs, s.vars.depvar_ops) ∪ get_depvars(pde.rhs, s.vars.depvar_ops)
        any(x -> unwrap_const(safe_unwrap(x)) isa Number, arguments(u)) && push!(terms, u)
    end
    return terms
end

"""
Numeric-arg depvars whose time slot is still `s.time`. Time-literals are
collected by `array_time_literal_terms` so `u(0, x)` is not read as `u(t, 0)`.
"""
function array_boundary_value_terms(pde, s)
    return [u_ for u_ in array_numeric_arg_depvars(pde, s) if !array_is_time_literal_term(u_, s)]
end

function array_time_literal_terms(pde, s)
    return [u_ for u_ in array_numeric_arg_depvars(pde, s) if array_is_time_literal_term(u_, s)]
end

function array_time_literal_has_spatial_edge(u_, s)
    for (c, a) in zip(s.args[operation(u_)], arguments(u_))
        isequal(c, s.time) && continue
        unwrap_const(safe_unwrap(a)) isa Number && return true
    end
    return false
end

"""
True when a time-literal sits under any `Differential`. `Dt` of a frozen field
is zero and would drop an initial-velocity residual; `Dx(u(0, x))` stays
fallback so this does not overlap `#664`.
"""
function array_has_time_literal_derivative(expr, s)
    expr = safe_unwrap(expr)
    iscall(expr) || return false
    op = operation(expr)
    if op isa Differential
        for a in arguments(expr)
            for v in get_depvars(a, s.vars.depvar_ops)
                array_is_time_literal_term(v, s) && return true
            end
            array_has_time_literal_derivative(a, s) && return true
        end
    end
    return any(a -> array_has_time_literal_derivative(a, s), arguments(expr))
end

"""
Equation-level checks for interior boundary values and time-literals. Throws
`ArrayFormFallback` when there is no slice form (staggered spatial edges,
edge-aligned mixed time+spatial edges, edge-aligned boundary-value derivatives,
non-IC times, off-edge samples).

Pure time-literals are valid on every grid alignment. Spatial boundary values
are accepted on `CenterAlignedGrid` and `EdgeAlignedGrid` (the latter via
`array_edge_aligned_boundary_value`). Mixed time+spatial literals require
`CenterAlignedGrid`.
"""
function array_validate_boundary_values(pde, s, derivweights)
    bvals = array_boundary_value_terms(pde, s)
    tlits = array_time_literal_terms(pde, s)
    isempty(bvals) && isempty(tlits) && return bvals
    G = get_grid_type(s)
    has_mixed_tlit = any(u_ -> array_time_literal_has_spatial_edge(u_, s), tlits)
    if !isempty(bvals) || has_mixed_tlit
        G <: StaggeredGrid && throw(
            ArrayFormFallback("boundary values on a staggered grid")
        )
        if G <: EdgeAlignedGrid
            has_mixed_tlit && throw(
                ArrayFormFallback(
                    "mixed time-literal and spatial edge on an edge-aligned grid"
                )
            )
            !isempty(array_boundary_value_derivative_terms(pde, s)) && throw(
                ArrayFormFallback(
                    "derivative of boundary value on an edge-aligned grid"
                )
            )
        elseif !(G <: CenterAlignedGrid)
            throw(
                ArrayFormFallback(
                    "boundary values in interior equations require a CenterAlignedGrid or EdgeAlignedGrid"
                )
            )
        end
    end
    (
        array_has_time_literal_derivative(pde.lhs, s) ||
            array_has_time_literal_derivative(pde.rhs, s)
    ) && throw(
        ArrayFormFallback("derivative of time-literal in interior equation")
    )
    for u_ in bvals
        u = depvar(u_, s)
        args = ivs(u, s)
        args_ = remove(arguments(u_), s.time)
        length(args_) == length(args) || throw(
            ArrayFormFallback(
                "boundary value $u_ has unexpected argument structure"
            )
        )
        for (j, a) in enumerate(args_)
            aval = unwrap_const(safe_unwrap(a))
            aval isa Number || continue
            array_boundary_edge_index(aval, args[j], s)
        end
    end
    t0_ic = s.vars.intervals[s.time][1]
    for u_ in tlits
        t0 = array_time_slot_literal(u_, s)
        isequal(t0, t0_ic) || throw(
            ArrayFormFallback("time-literal $u_ is not the initial time")
        )
        for (c, a) in zip(s.args[operation(u_)], arguments(u_))
            isequal(c, s.time) && continue
            aval = unwrap_const(safe_unwrap(a))
            aval isa Number || continue
            array_boundary_edge_index(aval, c, s)
        end
        array_resolved_ic_formula(depvar(u_, s), s, derivweights.icformulas)
    end
    return bvals
end

"""
Resolve a numeric boundary argument to the interpolation index `newindex` uses
with `shift = true` on an `EdgeAlignedGrid`: the lower domain edge is 1, the
upper is `n - 1` so `half_offset_centered_difference` sits on the midpoint
between that node and the next.

`array_boundary_edge_index` is the unshifted (`n`) form and must stay that way:
center-aligned views and any later derivative path index the last node directly.
"""
function array_edge_aligned_interp_index(xval, x, s)
    if isequal(xval, s.axies[x][1])
        return 1
    elseif isequal(xval, s.axies[x][end])
        return length(s, x) - 1
    else
        throw(
            ArrayFormFallback(
                "boundary value is not at a domain edge for $x = $xval"
            )
        )
    end
end

"""
Weights and tap indices of `derivweights.interpmap[x]` at the edge-aligned
interpolation index `II_j`, from `get_half_offset_weights_and_stencil` with
`bs = []` — the same call `boundary_value_maps` makes for `EdgeAlignedGrid`.
"""
function array_interp_weights_and_taps(u, x, II_j, s, derivweights)
    j = x2i(s, u, x)
    N = ndims(u, s)
    II = CartesianIndex(ntuple(k -> k == j ? II_j : 1, N))
    haskey(derivweights.interpmap, x) || throw(
        ArrayFormFallback("no interpolation operator for $x")
    )
    D = derivweights.interpmap[x]
    ws, Itap = try
        get_half_offset_weights_and_stencil(D, II, s, [], u, (j, x))
    catch e
        e isa InterruptException && rethrow(e)
        throw(ArrayFormFallback("could not build interpolation stencil for $u at $x"))
    end
    taps = [I[j] for I in Itap]
    ax = axes(s.discvars[u], j)
    all(t -> checkindex(Bool, ax, t), taps) || throw(
        ArrayFormFallback("interpolation taps for $u at $x fall outside the grid")
    )
    return collect(ws), taps
end

"""
The interpolated domain-edge value of `u_` on an `EdgeAlignedGrid`.

A single pinned argument (`u(t, 1)`, `u(t, 0, y)`) is the `interpmap`
stencil along that axis — a scalar when every spatial argument is fixed (or
the remaining ranges are length 1), a face of singleton-range taps otherwise.
Several pinned arguments (`u(t, 0, 0)`) are the tensor product of those
interpolators, which is the value at the domain corner the one-axis pointwise
map does not form.
"""
function array_edge_aligned_boundary_value(u_, s, ranges, indexmap, derivweights)
    u = depvar(u_, s)
    arr = array_variable(u, s)
    args = ivs(u, s)
    args_ = remove(arguments(u_), s.time)
    N = length(args_)
    length(args_) == length(args) || throw(
        ArrayFormFallback("boundary value $u_ has unexpected argument structure")
    )

    pin_dims = Int[]
    pin_index = Int[]
    pin_weights = Vector[]
    pin_taps = Vector{Int}[]
    dim_range = Vector{Any}(undef, N)
    for (j, a) in enumerate(args_)
        aval = unwrap_const(safe_unwrap(a))
        if aval isa Number
            II_j = array_edge_aligned_interp_index(aval, args[j], s)
            ws, taps = array_interp_weights_and_taps(u, args[j], II_j, s, derivweights)
            push!(pin_dims, j)
            push!(pin_index, II_j)
            push!(pin_weights, ws)
            push!(pin_taps, taps)
        else
            haskey(indexmap, args[j]) || throw(
                ArrayFormFallback(
                    "boundary value $u_ has a free argument not on the equation"
                )
            )
            dim_range[j] = ranges[indexmap[args[j]]]
        end
    end
    isempty(pin_dims) && throw(
        ArrayFormFallback("boundary value $u_ has no pinned spatial argument")
    )

    scalar_out = all(j -> (j in pin_dims) || length(dim_range[j]) == 1, 1:N)
    if scalar_out && length(pin_dims) == 1
        j = pin_dims[1]
        II = CartesianIndex(
            ntuple(k -> k == j ? pin_index[1] : first(dim_range[k]), N)
        )
        ufunc(v, I, _) = s.discvars[v][I]
        return half_offset_centered_difference(
            derivweights.interpmap[args[j]], II, s, [], (j, args[j]), u, ufunc
        )
    end

    np = length(pin_dims)
    weights = []
    slices = []
    for combo in Iterators.product(ntuple(i -> eachindex(pin_taps[i]), np)...)
        w = pin_weights[1][combo[1]]
        for k in 2:np
            w *= pin_weights[k][combo[k]]
        end
        rs = ntuple(N) do j
            pin_i = findfirst(isequal(j), pin_dims)
            if pin_i !== nothing
                t = pin_taps[pin_i][combo[pin_i]]
                return scalar_out ? t : (t:t)
            else
                r = dim_range[j]
                return scalar_out ? first(r) : r
            end
        end
        push!(weights, w)
        push!(slices, arr[rs...])
    end
    return scalar_out ? sym_dot(weights, slices) : array_stencil(weights, slices)
end

"""
The array element (every spatial argument fixed) or face slice (some arguments free
over the core box) corresponding to a boundary value like `u(t, 1)` or `u(t, 0, y)`.
On a center-aligned grid this is a direct index; on an edge-aligned grid it is the
`interpmap` interpolant at the domain edge. Fixed dimensions use a singleton `k:k`
range so the result broadcasts against the core slice; a fully-fixed reference is a
scalar, which broadcasts as well.
"""
function array_boundary_value_view(u_, s, ranges, indexmap, derivweights)
    get_grid_type(s) <: EdgeAlignedGrid && return array_edge_aligned_boundary_value(
        u_, s, ranges, indexmap, derivweights
    )
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
Substitution rules mapping each boundary value in `pde` to its array element, face
slice, or edge-aligned interpolant over the core box described by `ranges`. Call
after `array_validate_boundary_values`.

Returns a `Vector{<:Pair}` even when empty, so `vcat` into `ArrayifyContext.rules`
stays well-typed.
"""
function array_boundary_value_rules(pde, s, ranges, indexmap, derivweights)
    return Pair[
        safe_unwrap(u_) => array_boundary_value_view(
            u_, s, ranges, indexmap, derivweights
        )
            for u_ in array_boundary_value_terms(pde, s)
    ]
end

"""Interfaces for a boundary-value derivative in `x`: wrap if periodic, else `[]`."""
function array_boundary_derivative_bs(u, x, s, bcmap, periodic)
    haskey(periodic, x) || return []
    uop = operation(depvar(u, s))
    haskey(bcmap, uop) && haskey(bcmap[uop], x) || throw(
        ArrayFormFallback(
            "periodic boundary-value derivative missing interface map for $x"
        )
    )
    return filter_interfaces(bcmap[uop][x])
end

"""Index for `u_`: edge on pinned arguments, first core index on free ones."""
function array_boundary_value_index(u_, s, ranges, indexmap)
    u = depvar(u_, s)
    args = ivs(u, s)
    args_ = remove(arguments(u_), s.time)
    length(args_) == length(args) || throw(
        ArrayFormFallback("boundary value $u_ has unexpected argument structure")
    )
    return ntuple(length(args_)) do j
        aval = unwrap_const(safe_unwrap(args_[j]))
        if aval isa Number
            return array_boundary_edge_index(aval, args[j], s)
        end
        haskey(indexmap, args[j]) || throw(
            ArrayFormFallback("boundary value $u_ has a free argument not on the equation")
        )
        return first(ranges[indexmap[args[j]]])
    end
end

"""
`depvarderivbcmaps` stencil for `(Differential(x)^d)(u_)` as a scalar or face slice
over the core box.
"""
function array_boundary_value_derivative(
        x, d, u_, s, ranges, indexmap, derivweights, periodic, bcmap
    )
    u = depvar(u_, s)
    args = ivs(u, s)
    N = length(args)
    j = findfirst(isequal(x), args)
    j === nothing && throw(
        ArrayFormFallback("derivative direction $x is not an argument of $u")
    )
    Dop = get(derivweights.map, Differential(x)^d, nothing)
    Dop === nothing && throw(
        ArrayFormFallback("no centered operator for order $d in $x")
    )
    II0 = CartesianIndex(array_boundary_value_index(u_, s, ranges, indexmap))
    bs = array_boundary_derivative_bs(u, x, s, bcmap, periodic)
    stencil = try
        central_difference_weights_and_stencil(Dop, II0, s, bs, (j, x), u)
    catch e
        e isa InterruptException && rethrow(e)
        throw(
            ArrayFormFallback(
                "could not build boundary-value derivative stencil for order $d"
            )
        )
    end
    stencil == 0 && throw(
        ArrayFormFallback("boundary-value derivative of a 0-dimensional variable")
    )
    ws, Itap = stencil
    all(I -> all(k -> k == j || I[k] == II0[k], 1:N), Itap) ||
        throw(ArrayFormFallback("boundary-value derivative stencil is not axis aligned"))

    arr = array_variable(u, s)
    args_ = remove(arguments(u_), s.time)
    all_fixed = all(
        a -> unwrap_const(safe_unwrap(a)) isa Number, args_
    )
    slices = map(Itap) do I
        if all_fixed
            return arr[ntuple(k -> I[k], N)...]
        end
        rs = ntuple(N) do k
            aval = unwrap_const(safe_unwrap(args_[k]))
            if aval isa Number || k == j
                return I[k]:I[k]
            end
            haskey(indexmap, args[k]) || throw(
                ArrayFormFallback(
                    "boundary value $u_ has a free argument not on the equation"
                )
            )
            return ranges[indexmap[args[k]]]
        end
        return arr[rs...]
    end
    return array_stencil(collect(ws), slices)
end

"""Pointwise `depvarderivbcmaps` value for `(Differential(x)^d)(u_)` at this box."""
function array_point_boundary_derivative(
        x, d, u_, s, ranges, indexmap, derivweights, periodic, bcmap
    )
    u = depvar(u_, s)
    idxs = array_boundary_value_index(u_, s, ranges, indexmap)
    II = CartesianIndex(idxs)
    Dop = get(derivweights.map, Differential(x)^d, nothing)
    Dop === nothing && throw(
        ArrayFormFallback("no centered operator for order $d in $x")
    )
    bs = array_boundary_derivative_bs(u, x, s, bcmap, periodic)
    ufunc(v, I, _) = s.discvars[v][I]
    return central_difference(Dop, II, s, bs, (x2i(s, u, x), x), u, ufunc)
end

"""
Original and grid-valued keys for each handleable boundary-value derivative at
this singleton box. Applied before `expand_derivatives` in the pointwise path.
"""
function array_point_boundary_derivative_rules(
        pde, s, ranges, indexmap, derivweights, periodic, bcmap
    )
    rules = Pair[]
    for (expr, x, d, u_) in array_boundary_value_derivative_terms(pde, s)
        val = array_point_boundary_derivative(
            x, d, u_, s, ranges, indexmap, derivweights, periodic, bcmap
        )
        u = depvar(u_, s)
        args = ivs(u, s)
        args_ = remove(arguments(u_), s.time)
        idxs = array_boundary_value_index(u_, s, ranges, indexmap)
        gridsubs = Dict(
            safe_unwrap(args[j]) => s.grid[args[j]][idxs[j]]
                for (j, a) in enumerate(args_)
                if !(unwrap_const(safe_unwrap(a)) isa Number)
        )
        push!(rules, safe_unwrap(expr) => val)
        push!(rules, pde_substitute(expr, gridsubs) => val)
    end
    return rules
end

"""Rules mapping each handleable boundary-value derivative in `pde` to its stencil."""
function array_boundary_value_derivative_rules(
        pde, s, ranges, indexmap, derivweights, periodic, bcmap
    )
    return Pair[
        safe_unwrap(expr) => array_boundary_value_derivative(
            x, d, u_, s, ranges, indexmap, derivweights, periodic, bcmap
        )
            for (expr, x, d, u_) in array_boundary_value_derivative_terms(pde, s)
    ]
end

"""
Look up the order-0 IC formula for canonical `u`.
"""
function array_ic_formula_lookup(u, icformulas)
    ukey = safe_unwrap(u)
    haskey(icformulas, ukey) && return icformulas[ukey]
    for (k, v) in icformulas
        isequal(k, ukey) && return v
    end
    return nothing
end

"""
IC formula for `u`, with nested time-literals resolved.

`u(0,x) ~ v(0,x)` substitutes `v`'s formula. A fixed spatial argument is
bound in continuous coordinates (`v(0, 1)` becomes the formula at `x = 1`),
not replaced by the whole field of `v`. Remaining depvars, including live
fields left by `parse_bcs`, are replaced by that variable's IC.
"""
function array_resolved_ic_formula(u, s, icformulas, seen = nothing)
    # Path copy: `u ~ v(0, x) + v(0, 1)` visits `v` twice as siblings, not a cycle.
    seen = seen === nothing ? Any[] : copy(seen)
    ukey = safe_unwrap(u)
    formula = array_ic_formula_lookup(ukey, icformulas)
    formula === nothing && throw(
        ArrayFormFallback("no initial condition for time-literal of $u")
    )
    any(x -> isequal(x, ukey), seen) && throw(
        ArrayFormFallback("cyclic initial conditions involving $u")
    )
    push!(seen, ukey)
    expr = formula
    for v in get_depvars(expr, s.vars.depvar_ops)
        # `parse_bcs` may rewrite `v(0, x)` to `v(t, x)` and `v(0, 1)` to
        # `v(t, 1)`. Both are this variable's IC, with numeric spatial args
        # bound in continuous coordinates.
        inner = array_resolved_ic_formula(depvar(v, s), s, icformulas, seen)
        inner = array_bind_time_literal_spatial(v, s, inner)
        expr = pde_substitute(expr, Dict(v => inner))
    end
    return expr
end

"""
Bind numeric spatial arguments of time-literal `v` into `formula`.

The IC is a continuous expression, so `v(0, 1)` substitutes `x => 1`, not a
grid index. Free spatial arguments are left for the outer view to evaluate.
"""
function array_bind_time_literal_spatial(v, s, formula)
    subs = Dict()
    for (c, a) in zip(s.args[operation(v)], arguments(v))
        isequal(c, s.time) && continue
        aval = unwrap_const(safe_unwrap(a))
        aval isa Number || continue
        subs[safe_unwrap(c)] = aval
    end
    isempty(subs) && return formula
    return pde_substitute(formula, subs)
end

function array_ic_fold(v)
    v = unwrap_const(safe_unwrap(v))
    v isa Number && return float(v)
    return float(symbolic_to_float(v))
end

"""
Evaluate an IC formula at `spatial_subs` and the initial time.
"""
function array_ic_eval(formula, s, spatial_subs)
    subs = Dict{Any, Any}(spatial_subs)
    subs[safe_unwrap(s.time)] = s.vars.intervals[s.time][1]
    return array_ic_fold(pde_substitute(formula, subs))
end

"""
Spatial indices of a time-literal over `ranges`. Free arguments take the core
range (or its first index when `singleton`); fixed arguments are domain-edge
indices. The time slot is skipped.
"""
function array_time_literal_spatial_indices(u_, s, ranges, indexmap; singleton = false)
    idxs = []
    all_fixed = true
    coords = []
    for (c, a) in zip(s.args[operation(u_)], arguments(u_))
        isequal(c, s.time) && continue
        aval = unwrap_const(safe_unwrap(a))
        if aval isa Number
            i = array_boundary_edge_index(aval, c, s)
            push!(idxs, i)
            push!(coords, c)
        else
            all_fixed = false
            r = ranges[indexmap[c]]
            push!(idxs, singleton ? first(r) : r)
            push!(coords, c)
        end
    end
    return idxs, coords, all_fixed
end

"""
Evaluate the IC formula of `u_` on the spatial view. Arrays are one symbolic
term so `treesize` does not grow with the grid. Fixed spatial arguments become
length-1 axes to broadcast against the core, as in `array_boundary_value_view`.
"""
function array_time_literal_view(u_, s, ranges, indexmap, icformulas)
    formula = array_resolved_ic_formula(depvar(u_, s), s, icformulas)
    idxs, coords, all_fixed = array_time_literal_spatial_indices(u_, s, ranges, indexmap)
    if all_fixed
        subs = Dict(
            safe_unwrap(coords[j]) => s.grid[coords[j]][idxs[j]] for j in eachindex(coords)
        )
        return array_ic_eval(formula, s, subs)
    end
    rs = ntuple(j -> idxs[j] isa Integer ? (idxs[j]:idxs[j]) : idxs[j], length(idxs))
    out = Array{Float64}(undef, map(length, rs)...)
    for J in CartesianIndices(out)
        subs = Dict(
            safe_unwrap(coords[j]) => s.grid[coords[j]][rs[j][J[j]]]
                for j in eachindex(coords)
        )
        out[J] = array_ic_eval(formula, s, subs)
    end
    return array_hold_numeric(out)
end

function array_time_literal_rules(pde, s, ranges, indexmap, derivweights)
    return Pair[
        safe_unwrap(u_) => array_time_literal_view(
            u_, s, ranges, indexmap, derivweights.icformulas
        )
            for u_ in array_time_literal_terms(pde, s)
    ]
end

"""
Substitute boundary values and time-literals into an already pointwise-discretized
equation at the single point described by `ranges` (all singleton). Used for
size-1 wrap boxes and frame points, where `discretize_equation_at_point` leaves
interface-face and free-standing-corner boundary values symbolic. `valmaps` has
already replaced free spatial arguments with their grid values inside those
leftover terms (`u(t, 0, y)` appears as e.g. `u(t, 0, 0.2)`, `u(0, x)` as
`u(0, x_i)`), so each rule keys on that grid-valued form and maps to a scalar
array element (center-aligned) or interpolant (edge-aligned).

Boundary-value derivatives are already replaced via `extra_rules` before
`expand_derivatives`; this leftover sweep is for leftover values and time-literals.
A time-literal's time slot is not substituted; the value is the IC formula.
"""
function array_substitute_boundary_values(
        eq, pde, s, ranges, indexmap, derivweights, _periodic, _bcmap
    )
    bvals = array_boundary_value_terms(pde, s)
    tlits = array_time_literal_terms(pde, s)
    isempty(bvals) && isempty(tlits) && return eq
    rdict = Dict()
    for u_ in bvals
        u = depvar(u_, s)
        args = ivs(u, s)
        args_ = remove(arguments(u_), s.time)
        idxs = array_boundary_value_index(u_, s, ranges, indexmap)
        gridsubs = Dict(
            safe_unwrap(args[j]) => s.grid[args[j]][idxs[j]]
                for (j, a) in enumerate(args_)
                if !(unwrap_const(safe_unwrap(a)) isa Number)
        )
        val = get_grid_type(s) <: EdgeAlignedGrid ?
            array_edge_aligned_boundary_value(
                u_, s, ranges, indexmap, derivweights
            ) : array_variable(u, s)[idxs...]
        rdict[pde_substitute(u_, gridsubs)] = val
    end
    for u_ in tlits
        idxs, coords, _ = array_time_literal_spatial_indices(
            u_, s, ranges, indexmap; singleton = true
        )
        pointsubs = Dict(
            safe_unwrap(coords[j]) => s.grid[coords[j]][idxs[j]] for j in eachindex(coords)
        )
        gridsubs = Dict(
            safe_unwrap(c) => pointsubs[safe_unwrap(c)]
                for (c, a) in zip(s.args[operation(u_)], arguments(u_))
                if !isequal(c, s.time) && !(unwrap_const(safe_unwrap(a)) isa Number)
        )
        formula = array_resolved_ic_formula(depvar(u_, s), s, derivweights.icformulas)
        rdict[pde_substitute(u_, gridsubs)] = array_ic_eval(formula, s, pointsubs)
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
Per-slot coefficient arrays of a split functional scheme: `getcoeffs(i)` is evaluated once
per point of `rng` and its `nslots` entries sliced into broadcastable arrays along
dimension `j` of `N`, like `array_weight_vals`.
"""
function array_coeff_vals(getcoeffs, nslots, rng, j, N)
    percoeff = [getcoeffs(i) for i in rng]
    return map(1:nslots) do k
        vals = [c[k] for c in percoeff]
        N == 1 ? vals : reshape(vals, ntuple(i -> i == j ? length(vals) : 1, N))
    end
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
The interior branch of `central_difference_weights_and_stencil` for `Dop` along dimension
`j` of `N` over the index range `rng`, as `(weights, taps)`. On a nonuniform grid the
interior weights vary from point to point, so each returned weight is the broadcastable
numeric array of that tap's weight over `rng` rather than a scalar.
"""
function array_interior_stencil(Dop, rng, j, N)
    taps = half_range(Dop.stencil_length)
    weights = if Dop.dx isa Number
        collect(Dop.stencil_coefs)
    else
        bpc = Dop.boundary_point_count
        [
            array_weight_vals(i -> Dop.stencil_coefs[i - bpc], k, rng, j, N)
                for k in eachindex(taps)
        ]
    end
    return weights, taps
end

"""
Array form of `central_difference` on the core region for the even order derivative
`(Differential(x)^d)(u)`.
"""
function array_central_difference(Dop, s, u, x, d, ranges, indexmap, periodic)
    # `central_difference_weights_and_stencil` rejects interfaces on a nonuniform grid;
    # the pointwise path is the one that must report that.
    haskey(periodic, x) && !(Dop.dx isa Number) && throw(
        ArrayFormFallback("even-order derivative in a periodic nonuniform direction")
    )
    N = length(ranges)
    j = indexmap[x]
    weights, taps = array_interior_stencil(Dop, ranges[j], j, N)
    slices = [
        array_slice(u, s, ranges, indexmap; shiftx = x, offset = k, periodic = periodic)
            for k in taps
    ]
    return array_stencil(weights, slices)
end

@inline function array_cartesian_rules(
        s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
    )
    rules = Pair[]
    for u in depvars, x in ivs(depvar(u, s), s)
        haskey(pdeorders, x) || continue
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
The `(u, x, m, y, n)` records for which the mixed derivative
`(Differential(x)^m * Differential(y)^n)(u)` occurs in `pde`.

Two distinct spatial variables and a dependent variable is the only mixed family the
pointwise path has a scheme for (`generate_mixed_rules`), and so the only one with a
slice form here. Three-or-more spatial directions and mixed derivatives of anything
other than a dependent variable reach `arrayify` with a spatial differential still in
place and fall back.
"""
function array_mixed_terms(pde, depvars, args, pdeorders, s)
    found = []
    seen = []
    for u in depvars
        xs = filter(x -> array_iv_in(x, args), ivs(u, s))
        for x in xs, y in remove(xs, x)
            for mx in get(pdeorders, x, Int[]), my in get(pdeorders, y, Int[])
                keys = mixed_derivative_keys(u, x, mx, y, my)
                matched = false
                for key in keys
                    any(k -> isequal(k, key), seen) && continue
                    r = key => nothing
                    (subsmatch(pde.lhs, r) || subsmatch(pde.rhs, r)) || continue
                    push!(seen, key)
                    matched = true
                end
                matched || continue
                push!(found, (u, x, mx, y, my))
            end
        end
    end
    return found
end

"""
Array form of `mixed_central_difference` on the core region for
`(Differential(x)^m * Differential(y)^n)(u)`.

The scalar scheme is the tensor product of the two centered stencils: a sum over the
taps in `x` of a sum over the taps in `y` of `wx*wy*u[II + kx + ky]`. Every point of
the core takes the interior branch of both, so the whole thing is one broadcasted sum of
slices shifted along two axes at once — `array_central_difference` with a second shifted
axis. The weights come from the same `DerivativeOperator`s the pointwise path uses and their
products are numeric, so the two agree term by term. The first-order case `Dx(Dy(u))` is
`m = n = 1`.
"""
function array_mixed_difference(Dxop, Dyop, s, u, x, y, ranges, indexmap, periodic)
    # `central_difference_weights_and_stencil` rejects interfaces on a nonuniform grid;
    # the pointwise path is the one that must report that.
    haskey(periodic, x) && !(Dxop.dx isa Number) && throw(
        ArrayFormFallback("mixed derivative in a periodic nonuniform direction")
    )
    haskey(periodic, y) && !(Dyop.dx isa Number) && throw(
        ArrayFormFallback("mixed derivative in a periodic nonuniform direction")
    )
    N = length(ranges)
    jx = indexmap[x]
    jy = indexmap[y]
    xweights, xtaps = array_interior_stencil(Dxop, ranges[jx], jx, N)
    yweights, ytaps = array_interior_stencil(Dyop, ranges[jy], jy, N)
    weights = [broadcast(*, wx, wy) for wx in xweights for wy in yweights]
    slices = [
        array_shifted_slice(u, s, ranges, indexmap, Dict(jx => kx, jy => ky), periodic)
            for kx in xtaps for ky in ytaps
    ]
    return array_stencil(weights, slices)
end

@inline function array_mixed_rules(
        mixedterms, s, derivweights, ranges, indexmap, periodic
    )
    rules = Pair[]
    for (u, x, mx, y, my) in mixedterms
        expr = array_mixed_difference(
            derivweights.map[Differential(x)^mx],
            derivweights.map[Differential(y)^my],
            s, u, x, y, ranges, indexmap, periodic
        )
        for key in mixed_derivative_keys(u, x, mx, y, my)
            push!(rules, key => expr)
        end
    end
    return rules
end

"""
Patterns the staggered pointwise path cannot discretize either; falling back keeps this
strategy's behaviour identical to the pointwise form for them.
"""
function validate_staggered_array_form(s, depvars, pdeorders, args)
    for x in args
        all(isodd, pdeorders[x]) ||
            throw(ArrayFormFallback("even-order derivative on a staggered grid"))
        isempty(pdeorders[x]) && continue
        # the staggered pointwise path applies `stencil_coefs` directly, which only holds a
        # single weight set on a uniform grid
        s.dxs[x] isa Number ||
            throw(ArrayFormFallback("staggered grid with nonuniform d$x"))
    end
    for u in depvars
        ndims(u, s) == 0 && continue
        haskey(s.staggeredvars, operation(depvar(u, s))) ||
            throw(ArrayFormFallback("no alignment recorded for $u"))
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
        haskey(pdeorders, x) || continue
        for d in filter(isodd, pdeorders[x])
            Dop = get(derivweights.windmap[1], Differential(x)^d, nothing)
            Dop === nothing && throw(
                ArrayFormFallback(
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
    # `_upwind_difference` rejects interfaces on a nonuniform grid; the pointwise path is
    # the one that must report that.
    haskey(periodic, x) && !(Dop.dx isa Number) && throw(
        ArrayFormFallback("upwind derivative in a periodic nonuniform direction")
    )
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
`expr`, mirroring the pointwise path's `ifelse(coef > 0, coef*pos, coef*neg)`.

When the coefficient does not vary over the grid — a literal, a parameter, or any
expression of time alone — the wind direction is one scalar condition for the whole
slice, so `ifelse` broadcasts and reproduces the pointwise path exactly.

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
    array_scheme_split(F::FunctionalScheme)

Coefficient split of a functional scheme for nonuniform grids; `nothing` (the default)
falls back to the pointwise path there. Not exported: a scheme opts in by defining a
method on `MethodOfLines.array_scheme_split`, as WENO does.

A split `(coeffs, apply, nslots)` factors `interior` into grid geometry and solution
arithmetic: `coeffs(xwindow)` maps the `interior_points`-long window of grid coordinates
to `nslots` numbers, and `apply(u, p, t, c)` recombines them with the taps so that
`apply(u, p, t, coeffs(xwindow))` equals `interior(u, p, t, xwindow, dxwindow)` up to
reassociation. `apply` is traced once on placeholder symbols and must be branch free;
the slots are evaluated numerically per grid point. If each slot holds exactly the
constant the scalar trace folds at that spot and enters `apply` linearly, the array form
matches the pointwise path bitwise (WENO's split does, pinned by its property test); a
split that reassociates the fold agrees to ~1e-15 relative instead.
"""
array_scheme_split(::FunctionalScheme) = nothing

"""
Trace of a functional advection scheme for one direction `x`, shared by every variable
advected in `x`: the traced expression, tap offsets, placeholder taps `usyms`, and on
nonuniform grids the coefficient slots `csyms` with their split.

The interior is translation invariant, so the scheme traces once and its taps are later
replaced by shifted slices; solution-dependent weights (WENO's smoothness indicators) are
tap arithmetic and broadcast like any other term. Schemes that read the grid coordinate
fall back: the pointwise path folds those numeric coordinates, which a trace would rebuild
reassociated. Nonuniform grids trace the `apply` kernel of the scheme's
[`array_scheme_split`](@ref) instead; schemes without a split fall back. Periodic
nonuniform directions share the split kernel, with coefficient windows unwrapped across
the seam by `array_scheme_coeff_rules`.
"""
function array_function_scheme_trace(F, s, x, periodic)
    dx = s.dxs[x]
    # `get_f_taps_coords` rejects a stencil that wraps more than once around the seam
    (haskey(periodic, x) && periodic[x].n - 1 < F.interior_points) &&
        throw(ArrayFormFallback("too few points in $x for $(F.name) to wrap"))
    taps = half_range(F.interior_points)
    usyms = array_scheme_syms("u", length(taps))
    if dx isa Number
        xsyms = array_scheme_syms("x", length(taps))
        expr = try
            F.interior(usyms, vcat(F.ps, params(s)), Num(s.time), xsyms, dx)
        catch e
            e isa InterruptException && rethrow(e)
            throw(
                ArrayFormFallback(
                    "could not trace scheme $(F.name): $(sprint(showerror, e))"
                )
            )
        end
        any(
            v -> any(y -> isequal(v, safe_unwrap(y)), xsyms),
            Symbolics.get_variables(expr)
        ) && throw(
            ArrayFormFallback("scheme $(F.name) depends on the grid coordinate")
        )
        return (expr = expr, taps = taps, usyms = usyms, csyms = nothing, split = nothing)
    end
    split = array_scheme_split(F)
    split === nothing &&
        throw(ArrayFormFallback("$(F.name) advection on a nonuniform grid"))
    csyms = array_scheme_syms("c", split.nslots)
    expr = try
        split.apply(usyms, vcat(F.ps, params(s)), Num(s.time), csyms)
    catch e
        e isa InterruptException && rethrow(e)
        throw(
            ArrayFormFallback(
                "could not trace the coefficient split of scheme $(F.name): $(sprint(showerror, e))"
            )
        )
    end
    return (expr = expr, taps = taps, usyms = usyms, csyms = csyms, split = split)
end

"""
Coordinate of raw tap index `i` in a wrapping direction. Taps at or below the first
point take the lower-end destination chart, taps past the last point the upper-end
destination. Self-periodic (`dest === :self`) is a single add/subtract of the period;
a two-domain interface uses the partner grid and a contiguous (zero) shift when the
physical edges coincide. Mirrors `_wrapcoord` bit for bit — keep them in lockstep.

`array_periodic_coord` is the self-periodic special case.
"""
function array_wrap_coord(grid, i, spec, s)
    n = spec.n
    if i <= 1
        dest = spec.lower
        dest === nothing && return grid[i]
        grid′, n′ = if dest === :self
            grid, n
        else
            s.grid[dest.x], dest.n
        end
        return grid′[i + n′ - 1] - (grid′[end] - grid[1])
    elseif i > n
        dest = spec.upper
        dest === nothing && return grid[i]
        grid′ = dest === :self ? grid : s.grid[dest.x]
        return grid′[i + 1 - n] + (grid[end] - grid′[1])
    else
        return grid[i]
    end
end

function array_periodic_coord(grid, i, n)
    return array_wrap_coord(grid, i, (n = n, lower = :self, upper = :self), nothing)
end

"""
Rules binding the coefficient slots of a split scheme trace to their numeric per-point
arrays over the core box; empty on uniform grids. The windows fed to `coeffs` are the
same grid windows the pointwise path sees; in a periodic direction taps beyond either end
take the periodically shifted coordinate `bcoord` would produce.
"""
function array_scheme_coeff_rules(trace, s, x, ranges, indexmap, periodic)
    trace.csyms === nothing && return Pair[]
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]
    grid = s.grid[x]
    window = if haskey(periodic, x)
        spec = periodic[x]
        i -> [array_wrap_coord(grid, i + k, spec, s) for k in trace.taps]
    else
        i -> view(grid, i .+ trace.taps)
    end
    cvals = array_coeff_vals(
        i -> trace.split.coeffs(window(i)),
        trace.split.nslots, rng, j, N
    )
    return Pair[safe_unwrap(trace.csyms[k]) => cvals[k] for k in 1:trace.split.nslots]
end

"""
Array form of `function_scheme` on the interior branch, for the first derivative of `u`
in `x`: the shared per-direction trace with its taps replaced by shifted slices of `u`
and its coefficient slots, if any, by the numeric arrays in `crules`.
"""
function array_function_scheme(trace, crules, s, u, x, ranges, indexmap, periodic)
    slices = [
        array_slice(u, s, ranges, indexmap; shiftx = x, offset = k, periodic = periodic)
            for k in trace.taps
    ]
    rules = Pair[safe_unwrap(trace.usyms[k]) => slices[k] for k in eachindex(trace.taps)]
    return arrayify(trace.expr, ArrayifyContext(vcat(rules, crules), s.time))
end

@inline function array_advection_rules(
        pde, s, depvars, pdeorders, derivweights, ranges, indexmap, periodic
    )
    F = derivweights.advection_scheme
    F isa FunctionalScheme || return Pair[]
    # Only derivatives the equation contains: tracing absent pairs adds fallback surface.
    pairs = Tuple{Any, Any}[]
    for u in depvars, x in ivs(depvar(u, s), s)
        haskey(pdeorders, x) || continue
        1 in pdeorders[x] || continue
        key = safe_unwrap(Differential(x)(u))
        (subsmatch(pde.lhs, key => nothing) || subsmatch(pde.rhs, key => nothing)) ||
            continue
        push!(pairs, (u, x))
    end
    rules = Pair[]
    # One trace per direction: it depends on the grid along `x` but not on `u`.
    for x in unique(map(last, pairs))
        trace = array_function_scheme_trace(F, s, x, periodic)
        crules = array_scheme_coeff_rules(trace, s, x, ranges, indexmap, periodic)
        for (u, xu) in pairs
            isequal(xu, x) || continue
            push!(
                rules,
                safe_unwrap(Differential(x)(u)) => array_function_scheme(
                    trace, crules, s, u, x, ranges, indexmap, periodic
                )
            )
        end
    end
    return rules
end

@inline function array_winding_rules(
        terms, s, depvars, pdeorders, derivweights, ranges, indexmap, baserules, periodic
    )
    coefctx = ArrayifyContext(baserules, s.time)
    ruleobjs = []
    for u in depvars, x in ivs(depvar(u, s), s)
        haskey(pdeorders, x) || continue
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
        haskey(pdeorders, x) || continue
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
`terms`, keeping the last match per term (mirrors the pointwise path's `Dict` semantics).
Grid-varying prefactors throw: the pointwise path leaves them undiscretized, so no slice
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
                    ArrayFormFallback(
                        "grid-varying factor $(m.pre) multiplying a nonlinear laplacian has no slice form yet"
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
    wrap(i) = if haskey(periodic, x) && !array_wrap_is_twodomain(periodic[x])
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
in the coefficient surface as `ArrayFormFallback` from `arrayify`.
"""
function array_nonlinear_laplacian(
        m::NonlinlapMatch, s, depvars, derivweights, ranges, indexmap, periodic
    )
    x, u = m.x, m.u
    haskey(periodic, x) && !(s.dxs[x] isa Number) && throw(
        ArrayFormFallback("nonlinear laplacian in a periodic nonuniform direction")
    )
    if haskey(periodic, x) && array_wrap_is_twodomain(periodic[x]) &&
            array_mentions_iv(m.expr, x)
        throw(
            ArrayFormFallback(
                "nonlinear laplacian coefficient depends on $x across a two-domain interface"
            )
        )
    end
    N = length(ranges)
    j = indexmap[x]
    rng = ranges[j]

    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    interp = derivweights.interpmap[x]
    # Outer derivative is applied at II - 1 (`outerstencil` in the pointwise path).
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
        if !(haskey(periodic, x) && array_wrap_is_twodomain(periodic[x]))
            push!(
                rules,
                safe_unwrap(x) => array_interp_grid_vals(
                    interp, o, s, x, ranges, indexmap, periodic, N
                )
            )
        end
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
match are skipped: the pointwise path keys both rulesets by the same term and the nonlinear
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
                    ArrayFormFallback(
                        "grid-varying factor $(m.pre) multiplying a spherical laplacian has no slice form yet"
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

function array_shape_donor(rules)
    for r in rules
        is_array_valued(r.second) && return r.second
    end
    throw(ArrayFormFallback("no array slice to broadcast a scalar onto"))
end

function array_core_size(ranges)
    return ntuple(j -> length(ranges[j]), length(ranges))
end

function array_shape_donor(rules, ranges)
    target = array_core_size(ranges)
    for r in rules
        sl = r.second
        is_array_valued(sl) || continue
        Tuple(size(sl)) == target && return sl
    end
    return array_shape_donor(rules)
end

# `fill(val, shape)` is an `array_literal` of one node per point.
function array_broadcast_onto(val, ref)
    v = Symbolics.unwrap(val)
    v isa Number && iszero(v) && return broadcast(*, 0, ref)
    return broadcast(+, v, broadcast(*, 0, ref))
end

function array_integral_keys_equal(a, b)
    iscall(a) && iscall(b) || return false
    oa = operation(a)
    ob = operation(b)
    oa isa Integral && ob isa Integral || return false
    isequal(oa.domain.variables, ob.domain.variables) || return false
    da = oa.domain.domain
    db = ob.domain.domain
    isequal(safe_unwrap(da.left), safe_unwrap(db.left)) || return false
    isequal(safe_unwrap(da.right), safe_unwrap(db.right)) || return false
    return isequal(only(arguments(a)), only(arguments(b)))
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
    if iscall(expr) && operation(expr) isa Integral
        for (k, v) in ctx.rules
            array_integral_keys_equal(expr, k) && return v
        end
    end
    iscall(expr) || return Symbolics.wrap(expr)
    op = operation(expr)
    if op isa Differential
        isequal(op.x, ctx.time) ||
            throw(ArrayFormFallback("unhandled spatial derivative in $expr"))
        arg = arrayify(only(arguments(expr)), ctx)
        return op(arg)
    elseif !(op isa Function)
        # Symbolic operators (`Integral`, ...) cannot be broadcast over slices.
        throw(ArrayFormFallback("unhandled operation $op in $expr"))
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

Throws `ArrayFormFallback` for boundaries with no slice representation, in
which case the caller emits the pointwise form.
"""
function array_bc_eqs(s, boundary, interiormap, derivweights, bcmap)
    boundary isa AbstractTruncatingBoundary ||
        throw(ArrayFormFallback("non-truncating (interface) boundary"))

    u_, x_ = getvars(boundary)
    u = depvar(u_, s)
    args = ivs(u, s)
    length(args) == 0 && throw(ArrayFormFallback("no spatial arguments"))
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    haskey(indexmap, x_) ||
        throw(ArrayFormFallback("boundary variable $x_ not an argument of $u"))
    j = indexmap[x_]

    E = edge(s, boundary, interiormap)
    length(E) == 0 && throw(ArrayFormFallback("empty boundary edge"))
    lo = collect(Tuple(first(E)))
    hi = collect(Tuple(last(E)))
    # the face must be a contiguous box for a slice to describe it
    length(E) == prod(hi .- lo .+ 1) ||
        throw(ArrayFormFallback("boundary edge is not a contiguous box"))
    lo[j] == hi[j] ||
        throw(ArrayFormFallback("boundary edge spans its own direction"))
    ranges = Dict(i => lo[i]:hi[i] for i in eachindex(lo))
    N = length(args)
    # A single-point face (every 1D boundary) has nothing to collapse; a one-element
    # slice equation would just be a more convoluted spelling of the scalar one.
    prod(length(ranges[i]) for i in 1:N) == 1 &&
        throw(ArrayFormFallback("single-point boundary"))
    # A staggered 1D boundary is always the single point above; multi-point staggered
    # faces would need the staggered stencil selection, which has no slice form here yet.
    get_grid_type(s) <: StaggeredGrid &&
        throw(ArrayFormFallback("staggered boundary face"))

    # Every dependent variable in the condition must be one this path can slice: either
    # the canonical variable, or a value on this same boundary.
    bcdepvars = get_depvars(boundary.eq.lhs, s.vars.depvar_ops) ∪
        get_depvars(boundary.eq.rhs, s.vars.depvar_ops)
    for v in bcdepvars
        vd = depvar(v, s)
        if isempty(ivs(vd, s))
            continue
        end
        array_ivs_subset(ivs(vd, s), args) ||
            throw(ArrayFormFallback("variable $v of differing dimensionality"))
        for (k, a) in enumerate(remove(arguments(v), s.time))
            unwrap_const(safe_unwrap(a)) isa Number || continue
            k == j || throw(
                ArrayFormFallback("boundary value of $v away from this boundary")
            )
        end
    end
    # An interface on this same end wraps stencil taps off the face. An interface
    # at the opposite end of `x_` does not: this face's one-sided stencil points inward.
    ifaces = filter_interfaces(bcmap[operation(u)][x_])
    any(b -> isupper(b) == isupper(boundary), ifaces) &&
        throw(ArrayFormFallback("interface boundary condition in $x_"))

    II0 = first(E)
    ufunc(v, I, x) = s.discvars[v][I]

    # Derivatives in the boundary direction: take the weights and taps the pointwise path
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
            throw(ArrayFormFallback("could not build boundary stencil for order $d"))
        end
        offsets = [I[j] - II0[j] for I in Itap]
        all(I -> all(k -> k == j || I[k] == II0[k], 1:N), Itap) ||
            throw(ArrayFormFallback("boundary stencil is not axis aligned"))
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

    # Face slice, aligned to the face axes. Time-only variables are scalars.
    varrules = Pair[]
    bc_us = Any[u]
    for v in bcdepvars
        vd = depvar(v, s)
        push!(bc_us, vd)
        if isempty(ivs(vd, s))
            push!(varrules, safe_unwrap(v) => array_scalar_discvar(vd, s))
        else
            push!(varrules, safe_unwrap(v) => array_slice(vd, s, ranges, indexmap))
        end
    end
    push!(varrules, safe_unwrap(u) => array_slice(u, s, ranges, indexmap))
    intrules = array_integral_rules(
        s, unique(bc_us), ranges, indexmap; bvar = x_
    )

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

    ctx = ArrayifyContext(vcat(derivrules, intrules, varrules, gridrules), s.time)
    lhs = arrayify(boundary.eq.lhs, ctx)
    rhs = arrayify(boundary.eq.rhs, ctx)
    if is_array_valued(lhs) && !is_array_valued(rhs)
        rhs = array_broadcast_onto(rhs, array_shape_donor(varrules, ranges))
    elseif !is_array_valued(lhs) && is_array_valued(rhs)
        lhs = array_broadcast_onto(lhs, array_shape_donor(varrules, ranges))
    elseif !is_array_valued(lhs) && !is_array_valued(rhs)
        throw(ArrayFormFallback("boundary condition has no discretizable terms"))
    end
    return [lhs ~ rhs]
end

"""
    array_bc_eqs(s, boundary::InterfaceBoundary, interiormap, derivweights, bcmap)

Equate the two faces an interface (periodic) boundary joins as a single array equation,
the slice form of the `disc1[II] ~ disc2[II + Ioffset]` the pointwise path emits per point.

As in the pointwise path only the lower boundary of the pair carries the equations; the upper
one repeats the same relation and contributes none.
"""
function array_bc_eqs(s, boundary::InterfaceBoundary, interiormap, derivweights, bcmap)
    isupper(boundary) && return Equation[]

    u = depvar(boundary.u, s)
    u2 = depvar(boundary.u2, s)
    args = ivs(u, s)
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    haskey(indexmap, boundary.x) ||
        throw(ArrayFormFallback("boundary variable $(boundary.x) not an argument of $u"))
    j = indexmap[boundary.x]
    N = length(args)

    E = edge(s, boundary, interiormap)
    length(E) == 0 && throw(ArrayFormFallback("empty boundary edge"))
    lo = collect(Tuple(first(E)))
    hi = collect(Tuple(last(E)))
    length(E) == prod(hi .- lo .+ 1) ||
        throw(ArrayFormFallback("boundary edge is not a contiguous box"))
    lo[j] == hi[j] ||
        throw(ArrayFormFallback("boundary edge spans its own direction"))
    ranges = Dict(i => lo[i]:hi[i] for i in eachindex(lo))
    length(E) == 1 &&
        throw(ArrayFormFallback("single-point interface boundary"))
    # 1D staggered interfaces are the single point above; multi-point staggered faces
    # are untested on the pointwise path, so decline rather than guess.
    get_grid_type(s) <: StaggeredGrid &&
        throw(ArrayFormFallback("staggered interface face"))

    arr2 = array_variable(u2, s)
    disc2 = s.discvars[u2]
    interface_layout_compatible(s, boundary, j) || throw(
        ArrayFormFallback(
            "interface $(boundary.eq) joins variables with incompatible layout"
        )
    )
    # the same index shift `generate_bc_eqs!` applies pointwise
    shift = length(s, boundary.x2) - 1
    rs2 = ntuple(i -> i == j ? (ranges[i] .+ shift) : ranges[i], N)
    all(i -> checkindex(Bool, axes(disc2, i), rs2[i]), 1:N) ||
        throw(ArrayFormFallback("interface slice falls outside $(u2)"))

    return [array_slice(u, s, ranges, indexmap) ~ arr2[rs2...]]
end

"""
    array_bc_eqs(s, boundary::HigherOrderInterfaceBoundary, interiormap, derivweights, bcmap)

Flux (or other higher-order) interface condition as one array equation over the face.
`boundary.u` is discretized on this edge; `boundary.u2` on the partner face at index 1.
A self-periodic flux reuses one discrete variable on both ends; each call is bound to
its own face. A 1D (single-point) face falls back.
"""
function array_bc_eqs(
        s, boundary::HigherOrderInterfaceBoundary, interiormap, derivweights, bcmap
    )
    u = depvar(boundary.u, s)
    u2 = depvar(boundary.u2, s)
    x_ = boundary.x
    x2 = boundary.x2
    args = ivs(u, s)
    args2 = ivs(u2, s)
    length(args) == length(args2) ||
        throw(ArrayFormFallback("interface joins variables of differing dimensionality"))
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    indexmap2 = Dict([args2[i] => i for i in 1:length(args2)])
    haskey(indexmap, x_) ||
        throw(ArrayFormFallback("boundary variable $x_ not an argument of $u"))
    j = indexmap[x_]
    j2 = x2i(s, u2, x2)
    interface_layout_compatible(s, boundary, j) || throw(
        ArrayFormFallback(
            "interface $(boundary.eq) joins variables with incompatible layout"
        )
    )
    N = length(args)

    E = edge(s, boundary, interiormap)
    length(E) == 0 && throw(ArrayFormFallback("empty boundary edge"))
    lo = collect(Tuple(first(E)))
    hi = collect(Tuple(last(E)))
    length(E) == prod(hi .- lo .+ 1) ||
        throw(ArrayFormFallback("boundary edge is not a contiguous box"))
    lo[j] == hi[j] ||
        throw(ArrayFormFallback("boundary edge spans its own direction"))
    ranges = Dict(i => lo[i]:hi[i] for i in eachindex(lo))
    length(E) == 1 &&
        throw(ArrayFormFallback("single-point interface boundary"))
    get_grid_type(s) <: StaggeredGrid &&
        throw(ArrayFormFallback("staggered interface face"))

    ranges2 = Dict(i => i == j2 ? (1:1) : ranges[i] for i in 1:N)
    disc2 = s.discvars[u2]
    ndims(disc2) == N ||
        throw(ArrayFormFallback("interface joins variables of differing dimensionality"))
    rs2 = ntuple(i -> ranges2[i], N)
    all(i -> checkindex(Bool, axes(disc2, i), rs2[i]), 1:N) ||
        throw(ArrayFormFallback("interface slice falls outside $(u2)"))

    bcdepvars = get_depvars(boundary.eq.lhs, s.vars.depvar_ops) ∪
        get_depvars(boundary.eq.rhs, s.vars.depvar_ops)
    for v in bcdepvars
        vd = depvar(v, s)
        isequal(vd, u) || isequal(vd, u2) ||
            throw(ArrayFormFallback("variable $v is not on this interface"))
    end

    function face_deriv(ud, xd, II0, rs, imap, d)
        Dop = get(derivweights.map, Differential(xd)^d, nothing)
        Dop === nothing && return nothing
        jd = imap[xd]
        ws, Itap = try
            central_difference_weights_and_stencil(Dop, II0, s, [], (jd, xd), ud)
        catch e
            e isa InterruptException && rethrow(e)
            throw(ArrayFormFallback("could not build boundary stencil for order $d"))
        end
        offsets = [I[jd] - II0[jd] for I in Itap]
        all(I -> all(k -> k == jd || I[k] == II0[k], 1:N), Itap) ||
            throw(ArrayFormFallback("boundary stencil is not axis aligned"))
        slices = [
            array_slice(ud, s, rs, imap; shiftx = xd, offset = o) for o in offsets
        ]
        return array_stencil(collect(ws), slices)
    end

    II0 = first(E)
    II0_2 = CartesianIndex(ntuple(i -> i == j2 ? 1 : first(ranges[i]), N))
    samevar = isequal(u, u2)
    # Partner = `boundary.u2`. Same discrete variable: distinguish by the lower-end argument.
    function on_partner(v)
        v = safe_unwrap(v)
        isequal(v, safe_unwrap(boundary.u2)) && return true
        isequal(depvar(v, s), u2) || return false
        samevar || return true
        for (a, x) in zip(remove(arguments(v), s.time), ivs(depvar(v, s), s))
            isequal(x, x2) || continue
            aval = unwrap_const(safe_unwrap(a))
            return aval isa Number && isequal(aval, first(s.axies[x2]))
        end
        return false
    end

    derivrules = Pair[]
    for d in get(derivweights.orders, x_, Int[])
        expr = face_deriv(u, x_, II0, ranges, indexmap, d)
        expr === nothing && continue
        for v in bcdepvars
            on_partner(v) && continue
            push!(derivrules, safe_unwrap((Differential(x_)^d)(v)) => expr)
        end
        samevar || push!(derivrules, safe_unwrap((Differential(x_)^d)(u)) => expr)
    end
    for d in get(derivweights.orders, x2, Int[])
        expr = face_deriv(u2, x2, II0_2, ranges2, indexmap2, d)
        expr === nothing && continue
        for v in bcdepvars
            on_partner(v) || continue
            push!(derivrules, safe_unwrap((Differential(x2)^d)(v)) => expr)
        end
        samevar || push!(derivrules, safe_unwrap((Differential(x2)^d)(u2)) => expr)
    end

    varrules = Pair[]
    for v in bcdepvars
        if on_partner(v)
            push!(varrules, safe_unwrap(v) => array_slice(u2, s, ranges2, indexmap2))
        else
            push!(varrules, safe_unwrap(v) => array_slice(u, s, ranges, indexmap))
        end
    end
    if !samevar
        push!(varrules, safe_unwrap(u) => array_slice(u, s, ranges, indexmap))
        push!(varrules, safe_unwrap(u2) => array_slice(u2, s, ranges2, indexmap2))
    end

    gridrules = Pair[]
    for x in args
        if isequal(x, x_)
            val = lo[j] == 1 ? first(s.axies[x]) : last(s.axies[x])
            push!(gridrules, safe_unwrap(x) => val)
        else
            push!(gridrules, safe_unwrap(x) => array_grid_vals(x, s, ranges, indexmap, N))
        end
    end
    for x in args2
        any(y -> isequal(y, x), args) && continue
        if isequal(x, x2)
            push!(gridrules, safe_unwrap(x) => first(s.axies[x]))
        else
            push!(gridrules, safe_unwrap(x) => array_grid_vals(x, s, ranges2, indexmap2, N))
        end
    end
    # x2 is typically a different IV from x_; give it the partner-face endpoint.
    any(p -> isequal(p.first, safe_unwrap(x2)), gridrules) || push!(
        gridrules, safe_unwrap(x2) => first(s.axies[x2])
    )

    ctx = ArrayifyContext(vcat(derivrules, varrules, gridrules), s.time)
    lhs = arrayify(boundary.eq.lhs, ctx)
    rhs = arrayify(boundary.eq.rhs, ctx)
    if is_array_valued(lhs) && !is_array_valued(rhs)
        rhs = array_broadcast_onto(rhs, array_shape_donor(varrules))
    elseif !is_array_valued(lhs) && is_array_valued(rhs)
        lhs = array_broadcast_onto(lhs, array_shape_donor(varrules))
    elseif !is_array_valued(lhs) && !is_array_valued(rhs)
        throw(ArrayFormFallback("boundary condition has no discretizable terms"))
    end
    isequal(safe_unwrap(lhs), safe_unwrap(rhs)) &&
        throw(ArrayFormFallback("higher-order interface condition collapsed to an identity"))
    return [lhs ~ rhs]
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
    N >= 2 || throw(ArrayFormFallback("no corner region below 2 dimensions"))
    interior = interiormap.I[interiormap.pde[u]]
    length(interior) == 0 && throw(ArrayFormFallback("empty interior"))
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
