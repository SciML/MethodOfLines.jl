"""
    build_centered_stencil_matrix(D, gridlen, bs, x)

Build a sparse stencil matrix `L` of size `gridlen × gridlen` such that `L * u`
applies the centered finite difference approximation for the derivative operator `D`
at every grid point. Boundary stencils are used near domain edges; interior stencils
are used elsewhere. Interface (periodic) boundaries are handled via index wrapping.

# Arguments
- `D::DerivativeOperator`: Contains stencil coefficients and boundary coefficients
- `gridlen::Int`: Number of grid points along the dimension
- `bs`: Interface boundary list (from `filter_interfaces`)
- `x`: The independent variable for this dimension
"""
function build_centered_stencil_matrix(
        D::DerivativeOperator{T, N, Wind, DX}, gridlen, bs, x
    ) where {T, N, Wind, DX <: Number}
    haslower, hasupper = haslowerupper(bs, x)
    half = div(D.stencil_length, 2)

    L = spzeros(T, gridlen, gridlen)

    for i in 1:gridlen
        if (i <= D.boundary_point_count) & !haslower
            # Lower boundary one-sided stencil
            weights = D.low_boundary_coefs[i]
            offset = 1 - i
            for k in 1:D.boundary_stencil_length
                col = i + k - 1 + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        elseif (i > gridlen - D.boundary_point_count) & !hasupper
            # Upper boundary one-sided stencil
            weights = D.high_boundary_coefs[gridlen - i + 1]
            offset = gridlen - i
            for k in 1:D.boundary_stencil_length
                col = i + k - D.boundary_stencil_length + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        else
            # Interior centered stencil (with periodic wrapping if needed)
            weights = D.stencil_coefs
            for (k, offset) in enumerate(half_range(D.stencil_length))
                col = i + offset
                if length(bs) > 0
                    # Periodic wrapping
                    col = mod1(col, gridlen)
                end
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        end
    end

    return L
end

"""
    build_centered_stencil_matrix(D, gridlen, bs, x)

Non-uniform grid variant: stencil coefficients vary per grid point.
"""
function build_centered_stencil_matrix(
        D::DerivativeOperator{T, N, Wind, DX}, gridlen, bs, x
    ) where {T, N, Wind, DX <: AbstractVector}
    @assert length(bs) == 0 "Interface boundary conditions are not yet supported for nonuniform dx dimensions."
    half = div(D.stencil_length, 2)

    L = spzeros(T, gridlen, gridlen)

    for i in 1:gridlen
        if i <= D.boundary_point_count
            # Lower boundary one-sided stencil
            weights = D.low_boundary_coefs[i]
            offset = 1 - i
            for k in 1:D.boundary_stencil_length
                col = i + k - 1 + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        elseif i > gridlen - D.boundary_point_count
            # Upper boundary one-sided stencil
            weights = D.high_boundary_coefs[gridlen - i + 1]
            offset = gridlen - i
            for k in 1:D.boundary_stencil_length
                col = i + k - D.boundary_stencil_length + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        else
            # Interior stencil (coefficients vary per point for non-uniform grids)
            weights = D.stencil_coefs[i - D.boundary_point_count]
            for (k, offset) in enumerate(half_range(D.stencil_length))
                col = i + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        end
    end

    return L
end

"""
    build_upwind_stencil_matrix(D, gridlen, bs, x, forward::Bool)

Build a sparse stencil matrix for an upwind derivative operator.

The `forward` parameter indicates the stencil direction:
- `forward=true`: forward-biased stencil (range `0:len-1`), boundary at upper end.
  Used with `windmap[1]` (applied when wind is negative, i.e. `ispositive=false`).
- `forward=false`: backward-biased stencil (range `-(len-1):0`), boundary at lower end.
  Used with `windmap[2]` (applied when wind is positive, i.e. `ispositive=true`).

In `CompleteUpwindDifference`, `D.boundary_point_count` stores the high boundary count
(`stencil_length - 1 - offside`) and `D.offside` stores the low boundary count.
"""
function build_upwind_stencil_matrix(
        D::DerivativeOperator{T, N, Wind, DX}, gridlen, bs, x, forward::Bool
    ) where {T, N, Wind, DX <: Number}
    haslower, hasupper = haslowerupper(bs, x)

    # D.offside = low_boundary_point_count
    # D.boundary_point_count = high_boundary_point_count
    low_bpc = D.offside
    high_bpc = D.boundary_point_count

    L = spzeros(T, gridlen, gridlen)

    for i in 1:gridlen
        if forward
            # Forward-biased stencil: interior range is 0:(stencil_length-1)
            # Boundary handling at upper end only
            if (i > gridlen - high_bpc) & !hasupper
                # Upper boundary one-sided stencil
                weights = D.high_boundary_coefs[gridlen - i + 1]
                offset = gridlen - i
                for k in 1:D.boundary_stencil_length
                    col = i + k - D.boundary_stencil_length + offset
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            else
                # Interior forward stencil
                weights = D.stencil_coefs
                for (k, offset) in enumerate(0:(D.stencil_length - 1))
                    col = i + offset
                    if length(bs) > 0
                        col = mod1(col, gridlen)
                    end
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            end
        else
            # Backward-biased stencil: interior range is -(stencil_length-1):0
            # Boundary handling at lower end only
            if (i <= low_bpc) & !haslower
                # Lower boundary one-sided stencil
                weights = D.low_boundary_coefs[i]
                offset = 1 - i
                for k in 1:D.boundary_stencil_length
                    col = i + k - 1 + offset
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            else
                # Interior backward stencil
                weights = D.stencil_coefs
                for (k, offset) in enumerate((-(D.stencil_length - 1)):0)
                    col = i + offset
                    if length(bs) > 0
                        col = mod1(col, gridlen)
                    end
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            end
        end
    end

    return L
end

"""
    build_stencil_matrices(s, depvars, derivweights, bcmap)

Build all stencil matrices for all dependent variables and derivative orders.
Returns a nested dictionary: `matrices[u][Differential(x)^d] => sparse matrix L`.

For centered (even order) derivatives, a single matrix is returned.
For upwind (odd order) derivatives, two matrices are returned:
`(L_pos, L_neg)` for positive and negative wind directions.
"""
function build_stencil_matrices(s, depvars, derivweights, bcmap)
    matrices = Dict()

    for u in depvars
        uop = operation(u)
        u_matrices = Dict()

        for x in ivs(u, s)
            gridlen = length(s, x)
            bs = filter_interfaces(bcmap[uop][x])

            # Centered (even order) derivatives
            for d in derivweights.orders[x]
                if iseven(d)
                    D_op = derivweights.map[Differential(x)^d]
                    L = build_centered_stencil_matrix(D_op, gridlen, bs, x)
                    u_matrices[Differential(x)^d] = L
                end
            end

            # Upwind (odd order) derivatives
            for d in derivweights.orders[x]
                if isodd(d) && haskey(derivweights.windmap[1], Differential(x)^d)
                    # windmap[1]: used when ispositive=false → forward-biased stencil
                    D_fwd = derivweights.windmap[1][Differential(x)^d]
                    L_fwd = build_upwind_stencil_matrix(D_fwd, gridlen, bs, x, true)

                    # windmap[2]: used when ispositive=true → backward-biased stencil
                    D_bwd = derivweights.windmap[2][Differential(x)^d]
                    L_bwd = build_upwind_stencil_matrix(D_bwd, gridlen, bs, x, false)

                    u_matrices[Differential(x)^d] = (L_fwd, L_bwd)
                end
            end
        end

        matrices[uop] = u_matrices
    end

    return matrices
end
