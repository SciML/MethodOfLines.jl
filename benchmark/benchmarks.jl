# AirspeedVelocity / PkgBenchmark entry point: defines `SUITE`.
# Sizes keep the two-revision CI job under ~1 hour; heavier sweeps via
# `build_weno_suite` kwargs.

using BenchmarkTools
using MethodOfLines

include(joinpath(@__DIR__, "weno", "grids.jl"))
include(joinpath(@__DIR__, "weno", "problems.jl"))
include(joinpath(@__DIR__, "weno", "suite.jl"))

const SUITE = build_weno_suite(;
    resolutions = (64, 128),
    interface_resolutions = (41,),
    discretize_resolutions = (64,),
    interface_discretize_resolutions = (41,),
)
