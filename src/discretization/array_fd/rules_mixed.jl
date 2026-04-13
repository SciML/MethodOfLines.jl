# --- Mixed derivative ArrayOp rules -----------------------------------------

"""
    _build_mixed_derivative_rules(ctx::ArrayOpContext)

Build FD rules for mixed cross-derivatives `(Dx * Dy)(u)` using the Cartesian
product of two 1D centred stencils.
"""
function _build_mixed_derivative_rules(ctx::ArrayOpContext)
    s            = ctx.s
    depvars      = ctx.depvars
    derivweights = ctx.derivweights
    indexmap     = ctx.indexmap
    _idxs        = ctx.idxs
    bases        = ctx.bases
    is_periodic  = ctx.is_periodic
    gl_vec       = ctx.gl_vec
    mixed_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = _ConstSR(u_raw)
        u_spatial = ivs(u, s)
        for x in u_spatial
            # Need order-1 centred operator for this dimension
            haskey(derivweights.map, Differential(x)) || continue
            Dx_op = derivweights.map[Differential(x)]
            x_is_uniform = Dx_op.dx isa Number
            x_offsets = collect(half_range(Dx_op.stencil_length))

            # For non-uniform: build weight matrix and Const-wrap it
            dim_x_local = indexmap[x]
            if x_is_uniform
                x_weights = Dx_op.stencil_coefs
            else
                x_bpc = Dx_op.boundary_point_count
                if is_periodic[dim_x_local]
                    x_wmat = _build_periodic_wmat(Dx_op, collect(s.grid[x]))
                else
                    x_wmat = _stencil_coefs_to_matrix(Dx_op)
                end
                x_wmat_c = _ConstSR(x_wmat)
            end

            for y in u_spatial
                isequal(x, y) && continue
                haskey(derivweights.map, Differential(y)) || continue
                Dy_op = derivweights.map[Differential(y)]
                y_is_uniform = Dy_op.dx isa Number
                y_offsets = collect(half_range(Dy_op.stencil_length))

                dim_y_local = indexmap[y]
                if y_is_uniform
                    y_weights = Dy_op.stencil_coefs
                else
                    y_bpc = Dy_op.boundary_point_count
                    if is_periodic[dim_y_local]
                        y_wmat = _build_periodic_wmat(Dy_op, collect(s.grid[y]))
                    else
                        y_wmat = _stencil_coefs_to_matrix(Dy_op)
                    end
                    y_wmat_c = _ConstSR(y_wmat)
                end

                dim_x = indexmap[x]
                dim_y = indexmap[y]

                # Double sum: Σ_i Σ_j wx[i] * wy[j] * u[... + x_off[i] + y_off[j] ...]
                mixed_expr = sum(enumerate(x_offsets)) do (kx, x_off)
                    sum(enumerate(y_offsets)) do (ky, y_off)
                        tap = _tap_expr(ctx, u_c, u_spatial, Dict{Any,Any}(x => x_off, y => y_off))

                        wx = if x_is_uniform
                            x_weights[kx]
                        else
                            x_pt = is_periodic[dim_x] ? _idxs[dim_x] + bases[dim_x] : _idxs[dim_x] + bases[dim_x] - x_bpc
                            Symbolics.wrap(x_wmat_c[kx, x_pt])
                        end

                        wy = if y_is_uniform
                            y_weights[ky]
                        else
                            y_pt = is_periodic[dim_y] ? _idxs[dim_y] + bases[dim_y] : _idxs[dim_y] + bases[dim_y] - y_bpc
                            Symbolics.wrap(y_wmat_c[ky, y_pt])
                        end

                        wx * wy * tap
                    end
                end
                push!(mixed_rules, (Differential(x) * Differential(y))(u) => mixed_expr)
            end
        end
    end
    return mixed_rules
end

