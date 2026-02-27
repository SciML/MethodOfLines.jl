"""
Array-level finite difference rule generation for `ArrayDiscretization`.

For simple centred-difference PDEs on uniform grids with only even-order
derivatives, stencil weights are identical at every interior point in each
dimension.  This module pre-computes them once and builds a single ArrayOp
expression using N symbolic index variables `_i1, _i2, ...` (one per spatial
dimension).  The ArrayOp is a genuine symbolic array operation that, when
ModelingToolkit supports native array compilation, will compile to a single
vectorized loop.  Until then, MTK's `flatten_equations` scalarizes it into
individual per-point equations, preserving correctness.

All other cases -- non-uniform grids, odd-order derivatives (upwind), nonlinear
Laplacians, spherical/mixed derivatives, WENO, etc. -- fall back to per-point
computation via `discretize_equation_at_point` from the scalar path, which
supports ALL scheme types.
"""

# --- stencil pre-computation ------------------------------------------------

"""
    StencilInfo

Pre-computed information for a particular derivative operator.
"""
struct StencilInfo
    D_op::DerivativeOperator     # full operator, needed for boundary stencils
    offsets::Vector{Int}         # half_range(stencil_length)
    is_uniform::Bool             # true if dx is a Number
end

"""
    precompute_stencils(s, depvars, derivweights)

Returns a `Dict` mapping `(u, x, d)` to a `StencilInfo` for every
(variable, spatial dim, even derivative order) triple.
"""
function precompute_stencils(s, depvars, derivweights)
    info = Dict{Any, StencilInfo}()
    for u in depvars
        for x in ivs(u, s)
            for d in derivweights.orders[x]
                iseven(d) || continue
                D_op = derivweights.map[Differential(x)^d]
                info[(u, x, d)] = StencilInfo(
                    D_op,
                    collect(half_range(D_op.stencil_length)),
                    D_op.dx isa Number
                )
            end
        end
    end
    return info
end

"""
    stencil_weights_and_taps(si, II, j, grid_len, haslower, hasupper)

Compute the stencil weights and tap-point offsets at index `II` in dimension
`j`.  Handles boundary proximity exactly like
`central_difference_weights_and_stencil` from the scalar path.

Returns `(weights, Itap)` where `Itap` is a vector of `CartesianIndex`.
"""
function stencil_weights_and_taps(si::StencilInfo, II, j, ndim, grid_len, haslower, hasupper)
    D = si.D_op
    I1 = unitindex(ndim, j)
    idx = II[j]

    if (idx <= D.boundary_point_count) & !haslower
        # Near lower boundary -- use one-sided stencil
        weights = D.low_boundary_coefs[idx]
        offset = 1 - idx
        Itap = [II + (k + offset) * I1 for k in 0:(D.boundary_stencil_length - 1)]
    elseif (idx > (grid_len - D.boundary_point_count)) & !hasupper
        # Near upper boundary -- use one-sided stencil
        weights = D.high_boundary_coefs[grid_len - idx + 1]
        offset = grid_len - idx
        Itap = [II + (k + offset) * I1 for k in (-D.boundary_stencil_length + 1):0]
    else
        # True interior -- use centred stencil
        if si.is_uniform
            weights = D.stencil_coefs
        else
            weights = D.stencil_coefs[idx - D.boundary_point_count]
        end
        Itap = [II + off * I1 for off in si.offsets]
    end
    return weights, Itap
end

# --- interior equation generation -------------------------------------------

"""
    generate_array_interior_eqs(s, depvars, pde, derivweights, bcmap, eqvar,
                                 indexmap, boundaryvalfuncs, interior_ranges)

Generate discretised interior equations.

For the centred-stencil region on N-D uniform grids with only even-order
derivatives, a single ArrayOp equation is produced: the PDE's RHS is expressed
as an ArrayOp parameterised by symbolic index variables, and the LHS is
`D(u_arrayop)` where `u_arrayop` is an ArrayOp over the corresponding grid
slice.  This ArrayOp structure is ready for future native array compilation
in ModelingToolkit; currently MTK scalarizes it to per-point equations.

Boundary-proximity interior points (the "frame" around the centred region)
fall back to per-point computation via `discretize_equation_at_point`.

All other cases (non-uniform grids, odd-order derivatives such as upwind
schemes, nonlinear Laplacians, etc.) fall back entirely to per-point
computation, which supports ALL scheme types.
"""
function generate_array_interior_eqs(
        s, depvars, pde, derivweights, bcmap, eqvar,
        indexmap, boundaryvalfuncs, interior_ranges
    )
    stencil_cache = precompute_stencils(s, depvars, derivweights)

    # -- determine whether the ArrayOp path can handle this PDE ---------------
    # Requirements: (1) uniform grids, (2) even-order derivatives only.
    # When any requirement is unmet we fall back to per-point scalar
    # discretisation which supports ALL scheme types (upwind, nonlinear
    # Laplacian, spherical, mixed derivatives, WENO, etc.).
    all_uniform = isempty(stencil_cache) ? true :
        all(si.is_uniform for si in values(stencil_cache))
    all_even_orders = all(
        all(iseven(d) for d in derivweights.orders[x])
        for u in depvars for x in ivs(u, s)
    )
    can_template = all_uniform && all_even_orders

    ndim = length(interior_ranges)

    if !can_template
        # Full per-point fallback: delegate every interior point to the
        # scalar path's discretize_equation_at_point.
        cart_ranges = Tuple(r[1]:r[2] for r in interior_ranges)
        interior = CartesianIndices(cart_ranges)
        return collect(vec(map(interior) do II
            discretize_equation_at_point(
                II, s, depvars, pde, derivweights, bcmap,
                eqvar, indexmap, boundaryvalfuncs
            )
        end))
    end

    # -- N-D ArrayOp path (uniform grid, all even-order derivatives) ----------
    lo_vec = [r[1] for r in interior_ranges]
    hi_vec = [r[2] for r in interior_ranges]
    eqvar_ivs = ivs(eqvar, s)
    gl_vec = [length(s, x) for x in eqvar_ivs]

    # Per-dimension maximum boundary_point_count on each side across all
    # (variable, spatial dim, derivative order) triples.
    max_lower_bpc = zeros(Int, ndim)
    max_upper_bpc = zeros(Int, ndim)
    for u in depvars
        for (_, x) in enumerate(ivs(u, s))
            eq_dim = indexmap[x]  # dimension index in eqvar's ordering
            bs = filter_interfaces(bcmap[operation(u)][x])
            haslower, hasupper = haslowerupper(bs, x)
            for d in derivweights.orders[x]
                iseven(d) || continue
                bpc = stencil_cache[(u, x, d)].D_op.boundary_point_count
                if !haslower
                    max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], bpc)
                end
                if !hasupper
                    max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], bpc)
                end
            end
        end
    end

    lo_centered = [max(lo_vec[d], max_lower_bpc[d] + 1) for d in 1:ndim]
    hi_centered = [min(hi_vec[d], gl_vec[d] - max_upper_bpc[d]) for d in 1:ndim]
    n_centered  = [max(0, hi_centered[d] - lo_centered[d] + 1) for d in 1:ndim]

    # -- per-point equations for boundary-proximity interior points -----------
    # These are the "frame" around the centred region: all interior points
    # that are NOT in the centred rectangle.
    full_interior = CartesianIndices(Tuple(lo_vec[d]:hi_vec[d] for d in 1:ndim))
    centered_nonempty = all(lo_centered[d] <= hi_centered[d] for d in 1:ndim)

    eqs_boundary = Equation[]
    for II in full_interior
        in_centered = centered_nonempty &&
            all(lo_centered[d] <= II[d] <= hi_centered[d] for d in 1:ndim)
        in_centered && continue
        push!(eqs_boundary, discretize_equation_at_point(
            II, s, depvars, pde, derivweights, bcmap,
            eqvar, indexmap, boundaryvalfuncs
        ))
    end

    # -- ArrayOp equation for centred-stencil region --------------------------
    eqs_centered = if centered_nonempty && all(n_centered .> 0)
        candidate, eq_first = _build_centered_arrayop(
            n_centered, lo_centered, s, depvars, pde, derivweights,
            stencil_cache, eqvar, indexmap
        )
        # Validate: the template only handles simple (Differential(x)^d)(u)
        # patterns (centered differences).  PDEs with nonlinear Laplacian,
        # spherical, mixed-derivative, or other compound derivative forms are
        # correctly handled by the scalar path but silently produce wrong
        # equations via the template.  Validate by comparing the first
        # instantiated equation against the scalar path for the same point.
        II_check = CartesianIndex(Tuple(lo_centered))
        eq_scalar = discretize_equation_at_point(
            II_check, s, depvars, pde, derivweights, bcmap,
            eqvar, indexmap, boundaryvalfuncs
        )
        if isequal(eq_first.lhs, eq_scalar.lhs) &&
           isequal(eq_first.rhs, eq_scalar.rhs)
            candidate
        else
            # Template doesn't match scalar path -- fall back to per-point
            # for the centred region as well.
            centered_rect = CartesianIndices(
                Tuple(lo_centered[d]:hi_centered[d] for d in 1:ndim)
            )
            collect(vec(map(centered_rect) do II
                discretize_equation_at_point(
                    II, s, depvars, pde, derivweights, bcmap,
                    eqvar, indexmap, boundaryvalfuncs
                )
            end))
        end
    else
        Equation[]
    end

    return collect(vcat(eqs_boundary, eqs_centered))
end

# --- ArrayOp construction for centred-stencil region ------------------------

"""
    _build_centered_arrayop(n_centered, lo_centered, s, depvars, pde,
                             derivweights, stencil_cache, eqvar, indexmap)

Build a single ArrayOp equation for the centred-stencil interior region.

Constructs FD and variable/grid substitution rules parameterised by N symbolic
indices from `SymbolicUtils.idxs_for_arrayop(SymReal)`.  The PDE RHS is
transformed once to produce an ArrayOp expression.  The LHS is constructed as
`D(u_arrayop)` where `u_arrayop` is an ArrayOp indexing the discretised
variable at the centred grid points.

Returns `(eqs, eq_first)` where `eqs` is a single-element vector containing
the ArrayOp equation, and `eq_first` is the scalar equation at the first
centred point (for validation against the scalar path).
"""
function _build_centered_arrayop(
        n_centered, lo_centered, s, depvars, pde, derivweights,
        stencil_cache, eqvar, indexmap
    )
    ndim = length(n_centered)
    _idxs_arr = SymbolicUtils.idxs_for_arrayop(SymbolicUtils.SymReal)
    _idxs = [_idxs_arr[d] for d in 1:ndim]
    bases = [lo_centered[d] - 1 for d in 1:ndim]

    # -- FD rules using symbolic indices --------------------------------------
    fd_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        for (_, x) in enumerate(u_spatial)
            for d in derivweights.orders[x]
                iseven(d) || continue
                si = stencil_cache[(u, x, d)]
                weights = si.D_op.stencil_coefs  # uniform, centred
                taps = map(si.offsets) do off
                    idx_exprs = map(u_spatial) do xv
                        eq_d = indexmap[xv]
                        base_expr = _idxs[eq_d] + bases[eq_d]
                        isequal(xv, x) ? base_expr + off : base_expr
                    end
                    Symbolics.wrap(u_c[idx_exprs...])
                end
                expr = sym_dot(weights, taps)
                push!(fd_rules, (Differential(x)^d)(u) => expr)
            end
        end
    end

    # -- Variable/grid rules using symbolic indices ---------------------------
    var_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        idx_exprs = [_idxs[indexmap[xv]] + bases[indexmap[xv]] for xv in u_spatial]
        push!(var_rules, u => Symbolics.wrap(u_c[idx_exprs...]))
    end
    eqvar_ivs = ivs(eqvar, s)
    for x in eqvar_ivs
        grid_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(collect(s.grid[x]))
        dim = indexmap[x]
        push!(var_rules, x => Symbolics.wrap(grid_c[_idxs[dim] + bases[dim]]))
    end

    # -- Build templates (once) -----------------------------------------------
    # PDEBase rearranges equations to `lhs - rhs ~ 0`, so pde.lhs typically
    # contains both the time derivative D(u) and the spatial terms -stencil,
    # while pde.rhs = 0.  Apply all substitution rules to both sides.
    rdict = Dict(vcat(fd_rules, var_rules))
    template_lhs = expand_derivatives(pde_substitute(pde.lhs, rdict))
    template_rhs = pde_substitute(pde.rhs, rdict)

    # -- Separate time derivative from spatial terms --------------------------
    # Construct the known time-derivative template: D(eqvar_disc[_i + base])
    eqvar_raw = Symbolics.unwrap(s.discvars[eqvar])
    eqvar_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(eqvar_raw)
    eqvar_idx_exprs = [_idxs[d] + bases[d] for d in 1:ndim]
    dt_template = Differential(s.time)(Symbolics.wrap(eqvar_c[eqvar_idx_exprs...]))

    # Since template_lhs = D(u_disc[...]) + spatial_terms and template_rhs = 0
    # (after rearrangement to lhs ~ 0 form), the spatial RHS is:
    #   spatial_rhs = dt_template - template_lhs + template_rhs
    # This extracts everything except the time derivative and moves it to RHS.
    spatial_rhs = dt_template - template_lhs + template_rhs

    # -- Wrap in ArrayOps -----------------------------------------------------
    ao_ranges = Dict(_idxs[d] => (1:1:n_centered[d]) for d in 1:ndim)

    # LHS: D applied OUTSIDE the ArrayOp (scalarize/substitute cannot
    # penetrate Differential nodes, but D(ArrayOp) scalarizes correctly).
    u_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _idxs, eqvar_c[eqvar_idx_exprs...], +, nothing, ao_ranges
    )
    lhs_wrapped = Differential(s.time)(Symbolics.wrap(u_ao))

    # RHS: the spatial stencil expression as an ArrayOp.
    rhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _idxs, Symbolics.unwrap(spatial_rhs), +, nothing, ao_ranges
    )

    # -- The single ArrayOp equation ------------------------------------------
    arrayop_eq = lhs_wrapped ~ Symbolics.wrap(rhs_ao)

    # -- Also produce first scalar equation for validation --------------------
    # Use pde_substitute (which penetrates Differential) to instantiate the
    # full template at the first centred point.
    sub_first = Dict(_idxs[d] => 1 for d in 1:ndim)
    lhs_first = pde_substitute(template_lhs, sub_first)
    rhs_first = pde_substitute(template_rhs, sub_first)
    eq_first = lhs_first ~ rhs_first

    return [arrayop_eq], eq_first
end
