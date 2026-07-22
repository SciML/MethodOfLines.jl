# BenchmarkGroup hierarchy. Layers: "kernel" (reconstruction kernels only), "rhs" (one
# in-place RHS evaluation), "solve" (fixed-dt SSPRK33, adaptivity excluded), "discretize"
# (symbolic discretization cost).

using BenchmarkTools
using MethodOfLines
using SciMLBase
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK

function build_kernel_suite()
    g = BenchmarkGroup()

    u = [1.3, 2.1, 1.7, 0.4, 0.9]
    p = [1.0e-6]

    # Range view: matches how MOL passes coordinates on uniform grids.
    xr = StepRangeLen(0.0, 0.1, 100)
    xu = @view xr[48:52]

    x_nu = [0.0, 0.11, 0.23, 0.42, 0.55]
    dx_nu = diff(x_nu)

    # Equispaced nodes through the NU path: isolates kernel overhead.
    x_uv = uniform_grid(0.0, 0.4, 5)
    dx_uv = diff(x_uv)

    g["uniform"] = @benchmarkable MethodOfLines.weno_f($u, $p, 0.0, $xu, 0.1)
    g["nonuniform_center"] = @benchmarkable MethodOfLines.weno_f($u, $p, 0.0, $x_nu, $dx_nu)
    g["nonuniform_on_uniform_data"] =
        @benchmarkable MethodOfLines.weno_f($u, $p, 0.0, $x_uv, $dx_uv)

    bnd = BenchmarkGroup()
    for T in (1, 2, 4, 5)
        b = MethodOfLines.WENONonUniformBoundary{T}()
        bnd["target_$T"] = @benchmarkable $b($u, $p, 0.0, $x_nu, $dx_nu)
    end
    g["boundary"] = bnd

    return g
end

function rhs_benchmark(prob)
    f = prob.f
    u0 = copy(prob.u0)
    du = similar(u0)
    p = prob.p
    return @benchmarkable $f($du, $u0, $p, 0.0)
end

function solve_benchmark(prob, dt)
    return @benchmarkable solve(
        $prob, SSPRK33(); dt = $dt, adaptive = false, save_everystep = false
    ) evals = 1
end

function discretize_benchmark(pdesys, disc)
    return @benchmarkable discretize($pdesys, $disc) seconds = 60 samples = 3 evals = 1
end

const CFL_TARGET = 0.4

function build_weno_suite(;
        resolutions = (64, 256),
        interface_resolutions = (41, 81),
        discretize_resolutions = (64,),
        interface_discretize_resolutions = (41,),
    )
    suite = BenchmarkGroup()
    suite["kernel"] = build_kernel_suite()

    rhs = suite["rhs"] = BenchmarkGroup()
    slv = suite["solve"] = BenchmarkGroup()
    dsc = suite["discretize"] = BenchmarkGroup()

    single_domain = (
        ("advection", advection_system(), advection_discretization, 1.0),
        ("burgers", burgers_system(), burgers_discretization, 1.3),
    )

    for (label, sys, mkdisc, wavespeed) in single_domain
        rhs[label] = BenchmarkGroup()
        slv[label] = BenchmarkGroup()
        dsc[label] = BenchmarkGroup()
        for kind in GRID_KINDS
            kstr = String(kind)
            rhs[label][kstr] = BenchmarkGroup()
            slv[label][kstr] = BenchmarkGroup()
            dsc[label][kstr] = BenchmarkGroup()
            for n in resolutions
                prob = discretize(sys.pdesys, mkdisc(sys, kind, n))
                rhs[label][kstr]["N=$n"] = rhs_benchmark(prob)
                dt = CFL_TARGET * min_spacing(kind, sys.xspan..., n) / wavespeed
                slv[label][kstr]["N=$n"] = solve_benchmark(prob, dt)
            end
            for n in discretize_resolutions
                dsc[label][kstr]["N=$n"] = discretize_benchmark(sys.pdesys, mkdisc(sys, kind, n))
            end
        end
    end

    itf = interface_system()
    for group in (rhs, slv, dsc)
        group["interface"] = BenchmarkGroup()
    end
    for kind in (:uniform_vector, :stretched)
        kstr = String(kind)
        rhs["interface"][kstr] = BenchmarkGroup()
        slv["interface"][kstr] = BenchmarkGroup()
        dsc["interface"][kstr] = BenchmarkGroup()
        for n in interface_resolutions
            prob = discretize(itf.pdesys, interface_discretization(itf, kind, n))
            rhs["interface"][kstr]["N=$n"] = rhs_benchmark(prob)
            dt = CFL_TARGET * interface_min_spacing(kind, n)
            slv["interface"][kstr]["N=$n"] = solve_benchmark(prob, dt)
        end
        for n in interface_discretize_resolutions
            dsc["interface"][kstr]["N=$n"] =
                discretize_benchmark(itf.pdesys, interface_discretization(itf, kind, n))
        end
    end

    return suite
end
