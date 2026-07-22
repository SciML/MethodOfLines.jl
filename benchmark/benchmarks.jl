# PkgBenchmark / AirspeedVelocity entry point: defines `SUITE`.
# ENV["MOL_BENCH_MODE"] ∈ ("smoke", "default", "full") selects the resolution sweep.

using BenchmarkTools
using MethodOfLines

include(joinpath(@__DIR__, "weno", "grids.jl"))
include(joinpath(@__DIR__, "weno", "problems.jl"))
include(joinpath(@__DIR__, "weno", "suite.jl"))

const BENCH_MODE = get(ENV, "MOL_BENCH_MODE", "default")

const BENCH_SIZES = if BENCH_MODE == "smoke"
    (;
        resolutions = (32,),
        interface_resolutions = (21,),
        discretize_resolutions = (32,),
        interface_discretize_resolutions = (21,),
    )
elseif BENCH_MODE == "full"
    (;
        resolutions = (64, 128, 256, 512),
        interface_resolutions = (41, 81, 161),
        discretize_resolutions = (64, 128, 256),
        interface_discretize_resolutions = (41, 81),
    )
else
    (;
        resolutions = (64, 256),
        interface_resolutions = (41, 81),
        discretize_resolutions = (64,),
        interface_discretize_resolutions = (41,),
    )
end

const SUITE = build_weno_suite(; BENCH_SIZES...)
