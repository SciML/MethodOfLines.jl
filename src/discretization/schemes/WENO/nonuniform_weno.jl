@noinline function weno_f_nonuniform(u, p, t, x, dx::AbstractVector)
    throw(ArgumentError("WENO on non-uniform grids is not yet implemented."))
end

Base.@propagate_inbounds @inline weno_f_nonuniform(u, p, t, x, dx::Real) =
    weno_f_uniform(u, p, t, x, dx)
