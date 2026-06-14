using Test
using MethodOfLines

const weno_f = MethodOfLines.weno_f
const weno_f_uniform = MethodOfLines.weno_f_uniform

@testset "WENO dispatch — MOL-realistic stencil" begin
    u = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = [1.0e-6]
    t = 0.0
    dx_scalar = 0.1
    x = StepRangeLen(0.0, 0.1, 100)
    discx = @view x[[48, 49, 50, 51, 52]]
    @test weno_f(u, p, t, discx, dx_scalar) ≈ weno_f_uniform(u, p, t, discx, dx_scalar)
    @test @allocated(weno_f(u, p, t, discx, dx_scalar)) == 0
end

@testset "WENO dispatch — non-uniform stub" begin
    u = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = [1.0e-6]
    t = 0.0
    x_vec = collect(0.0:0.1:0.4)
    xv = @view x_vec[1:5]
    dx_vec = diff(x_vec[1:5])
    @test_throws ArgumentError("WENO on non-uniform grids is not yet implemented.") weno_f(
        u, p, t, xv, dx_vec
    )
end

@testset "WENO dispatch — scalar fallback on vector grid" begin
    u = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = [1.0e-6]
    t = 0.0
    dx_scalar = 0.1
    xv = collect(0.0:0.1:0.4)
    @test weno_f(u, p, t, xv, dx_scalar) ≈ weno_f_uniform(u, p, t, xv, dx_scalar)
end

@testset "WENO dispatch — unsupported dx type" begin
    u = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = [1.0e-6]
    t = 0.0
    x = 0.0:0.1:0.4
    @test_throws ArgumentError(
        "WENO expects dx to be a scalar (uniform) or AbstractVector (non-uniform); got String."
    ) weno_f(u, p, t, x, "0.1")
    @test_throws ArgumentError(
        "WENO expects dx to be a scalar (uniform) or AbstractVector (non-uniform); got Matrix{Float64}."
    ) weno_f(u, p, t, x, fill(0.1, 2, 2))
end
