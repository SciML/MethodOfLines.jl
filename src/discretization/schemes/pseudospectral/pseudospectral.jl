# Pseudospectral discretization.
#
# A spectral differentiation matrix is used in place of the finite difference
# stencils. Each row of the matrix is the full-width stencil for one grid
# point, so the same `central_difference`/`sym_dot` machinery as the finite
# difference path applies: the taps of an interior point are simply all the
# collocation points of the direction.
#
# For a periodic (Fourier) direction the grid holds `N + 1` points for `N`
# distinct equispaced collocation points - the first and last grid points are
# the same physical node. `rowmap` folds the duplicated endpoint back to the
# first distinct point and `taps` runs over the distinct points `2:n` only, so
# the algebraic boundary unknown `u[1]` (pinned by `u[1] ~ u[end]`) is never
# tapped and never counted twice.

"""
    SpectralDerivativeOperator

Dense differentiation matrix acting along one independent variable. `mat[i, :]`
is the differentiation row for matrix-index `i`; `rowmap[g]` maps grid index
`g` to its matrix row; `taps` lists the grid indices the columns act on.
"""
struct SpectralDerivativeOperator{T <: Real, M <: AbstractMatrix{T}}
    derivative_order::Int
    mat::M
    taps::Vector{Int}
    rowmap::Vector{Int}
end

"""
The `DifferentialDiscretizer` analogue for `PseudospectralDiscretization`: maps
`Differential(x)^d` to the corresponding `SpectralDerivativeOperator`.
"""
struct SpectralDifferentialDiscretizer{D1} <: AbstractDifferentialDiscretizer
    map::D1
    orders::Any
    icformulas::Any
end

"""
    spectral_grid(spec, a, b)

The collocation grid for `spec` on the domain `[a, b]`.
"""
function spectral_grid(spec::ChebyshevCollocation, a, b)
    n = spec.n
    return [a + (b - a) / 2 * (1 - cospi((k - 1) / (n - 1))) for k in 1:n]
end

function spectral_grid(spec::FourierCollocation, a, b)
    return collect(range(a, b, spec.n + 1))
end

function spectral_grid(spec::AbstractVector, a, b)
    issorted(spec) || throw(ArgumentError("Custom collocation grids must be sorted in ascending order."))
    (spec[1] ≈ a && spec[end] ≈ b) ||
        throw(ArgumentError("Custom collocation grid endpoints $(spec[1]), $(spec[end]) do not match the domain [$a, $b]."))
    return collect(spec)
end

"""
    spectral_diff_matrix(spec, grid, order)

Differentiation matrix of order `order` for the collocation scheme `spec` on
`grid`. Fourier directions use the explicit trigonometric-interpolant first
derivative matrix (Trefethen, *Spectral Methods in MATLAB*) raised to `order`;
everything else uses the maximal-stencil Fornberg weights, i.e. the
polynomial-interpolant differentiation matrix.
"""
function spectral_diff_matrix(spec, grid, order)
    n = length(grid)
    order < n ||
        throw(ArgumentError("Cannot take derivative order $order on $n collocation points."))
    D = zeros(eltype(grid), n, n)
    for i in 1:n
        D[i, :] = calculate_weights(order, grid[i], grid)
    end
    return D
end

function spectral_diff_matrix(spec::FourierCollocation, grid, order)
    N = spec.n
    L = grid[end] - grid[1]
    D = zeros(eltype(grid), N, N)
    for i in 1:N, j in 1:N
        i == j && continue
        k = i - j
        if iseven(N)
            D[i, j] = (π / L) * (-1)^k * cot(π * k / N)
        else
            D[i, j] = (π / L) * (-1)^k * csc(π * k / N)
        end
    end
    return D^order
end

"""
    spectral_taps_rowmap(spec, n)

The tapped grid indices and the grid-index-to-matrix-row map for `spec` on a
grid of `n` points.
"""
spectral_taps_rowmap(spec, n) = collect(1:n), collect(1:n)
function spectral_taps_rowmap(spec::FourierCollocation, n)
    # Interior indices are 2:n; matrix index j pairs with grid index j + 1.
    # Grid index 1 aliases grid index n, i.e. matrix index n - 1 = spec.n.
    return collect(2:n), [mod(i - 2, spec.n) + 1 for i in 1:n]
end

function PDEBase.construct_differential_discretizer(
        pdesys, s::DiscreteSpace, discretization::PseudospectralDiscretization, orders
    )
    differentialmap = Dict{Differential, SpectralDerivativeOperator}()
    for x in s.x̄
        spec = discretization.dxs[x]
        grid = s.grid[x]
        n = length(grid)
        taps, rowmap = spectral_taps_rowmap(spec, n)
        for d in union(orders[x], [1])
            D = spectral_diff_matrix(spec, grid, d)
            differentialmap[Differential(x)^d] = SpectralDerivativeOperator(
                d, D, taps, rowmap
            )
        end
    end
    return SpectralDifferentialDiscretizer{typeof(differentialmap)}(
        differentialmap, orders, array_ic_formulas(pdesys, s)
    )
end

"""
`central_difference_weights_and_stencil` for `SpectralDerivativeOperator`:
the stencil of an interior point is the full matrix row over all collocation
points of the direction.
"""
function central_difference_weights_and_stencil(
        D::SpectralDerivativeOperator, II, s, bs, jx, u
    )
    j, x = jx
    ndims(u, s) == 0 && return 0
    I1 = unitindex(ndims(u, s), j)
    weights = D.mat[D.rowmap[II[j]], :]
    Itap = [II + (k - II[j]) * I1 for k in D.taps]
    return weights, Itap
end

"""
    spectral_eval(ex, II, s, derivweights, indexmap)

Evaluate the expression `ex` at grid point `II` under spectral discretization.
Spatial differentials become differentiation-matrix row dots; dependent
variable calls become the matching discrete unknown (boundary literals resolve
to the boundary index via `newindex`); independent variables become their grid
value. This is the collocation analogue of the substitution rules used by the
finite difference path, and transparently handles nested derivative terms such
as `Dx(a(u) * Dx(u))` by evaluating the argument at every tap of the outer
derivative row.
"""
function spectral_eval(ex, II, s, derivweights, indexmap)
    ex = safe_unwrap(ex)
    if !iscall(ex)
        for (x, j) in indexmap
            isequal(ex, safe_unwrap(x)) && return s.grid[x][II[j]]
        end
        return ex
    end
    op = operation(ex)
    args = arguments(ex)
    if op isa Differential
        x = op.x
        if s.time !== nothing && isequal(safe_unwrap(x), safe_unwrap(s.time))
            return op(spectral_eval(args[1], II, s, derivweights, indexmap))
        end
        haskey(indexmap, x) || throw(
            ArgumentError("Cannot take a spectral derivative with respect to $x, which is not an independent variable of the discretized equation.")
        )
        D = derivweights.map[Differential(x)^op.order]
        return spectral_apply(D, args[1], II, s, derivweights, indexmap, x)
    elseif op isa Integral
        throw(ArgumentError("PseudospectralDiscretization does not yet support `Integral` terms, got $ex. Please post an issue if you need this feature."))
    elseif any(o -> isequal(op, o), s.vars.depvar_ops)
        u = depvar(ex, s)
        return s.discvars[u][newindex(ex, II, s, indexmap)]
    else
        return op(Tuple(spectral_eval(a, II, s, derivweights, indexmap) for a in args)...)
    end
end

"""
    spectral_apply(D, inner, II, s, derivweights, indexmap, x)

`Differential(x)^d(inner)` at `II` as a differentiation-matrix row dot. When
`inner` is a dependent variable call the literal arguments pin the row (so
`Dx(u(t, a))` is the derivative at the boundary), otherwise `inner` is
recursively evaluated at every tap of the row.
"""
function spectral_apply(D, inner, II, s, derivweights, indexmap, x)
    if iscall(inner) && any(o -> isequal(operation(inner), o), s.vars.depvar_ops)
        u = depvar(inner, s)
        p = x2i(s, u, x)
        # `inner` does not depend on `x`: its derivative vanishes
        p === nothing && return 0
        base = Tuple(newindex(inner, II, s, indexmap))
        row = D.mat[D.rowmap[base[p]], :]
        vals = map(D.taps) do k
            idx = CartesianIndex(base[1:(p - 1)]..., k, base[(p + 1):end]...)
            s.discvars[u][idx]
        end
        return sym_dot(row, vals)
    else
        j = indexmap[x]
        I1 = unitindex(length(II), j)
        row = D.mat[D.rowmap[II[j]], :]
        vals = map(D.taps) do k
            spectral_eval(inner, II + (k - II[j]) * I1, s, derivweights, indexmap)
        end
        return sym_dot(row, vals)
    end
end

"""
`generate_finite_difference_rules` for `SpectralDifferentialDiscretizer`.

Emits a substitution rule for every `Differential(x)^d)(u)` appearing, plus a
rule for each term headed by a spatial derivative whose argument is not a
plain dependent variable call (nested derivatives such as `Dx(u * Dx(u))`,
mixed derivatives `Dx(Dy(u))`, and boundary-value derivatives like
`Dx(u(t, 0))`).
"""
function generate_finite_difference_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, pde::Equation,
        derivweights::SpectralDifferentialDiscretizer, bmap, indexmap
    )
    length(II) == 0 && return []
    terms = split_terms(pde, s.x̄)
    rules = Pair[]
    stencilvars = idx_depvars(depvars, s, indexmap)
    for u in stencilvars
        for x in ivs(u, s)
            for d in derivweights.orders[x]
                term = (Differential(x)^d)(u)
                push!(
                    rules,
                    term => spectral_eval(term, II, s, derivweights, indexmap)
                )
            end
        end
    end
    # Rules for spatial-derivative-headed subterms not covered by the depvar
    # rules above: nested derivatives (`Dx(u * Dx(u))`, `Dx(u^2)`), mixed
    # derivatives, and derivatives of literal boundary values. Diff-headed
    # subterms are evaluated by `spectral_eval`, which handles any deeper
    # nesting itself, so there is no need to descend below them.
    diff_terms = Any[]
    for t in terms
        find_diff_terms!(diff_terms, t, s.x̄)
    end
    for t in unique!(diff_terms)
        push!(rules, t => spectral_eval(t, II, s, derivweights, indexmap))
    end
    return rules
end

function find_diff_terms!(out, t, x̄)
    iscall(t) || return
    op = operation(t)
    if op isa Differential && any(x -> isequal(op.x, x), x̄)
        push!(out, t)
        return
    elseif op isa Integral
        throw(ArgumentError("PseudospectralDiscretization does not yet support `Integral` terms, got $t. Please post an issue if you need this feature."))
    end
    for a in arguments(t)
        find_diff_terms!(out, a, x̄)
    end
    return
end
