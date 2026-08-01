# MethodOfLines.jl WENO Benchmarks

Benchmark suite comparing the non-uniform WENO-5 implementation against its uniform
counterpart. `benchmark/benchmarks.jl` defines `const SUITE :: BenchmarkGroup`, the
standard entry point consumed by
[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl).

## Layout

```
benchmark/
  Project.toml   environment for local runs (not used by the CI action)
  benchmarks.jl  SUITE definition (AirspeedVelocity entry point)
  weno/
    grids.jl     grid classes
    problems.jl  PDE definitions (advection, Burgers, two-domain interface)
    suite.jl     BenchmarkGroup hierarchy (kernel / rhs / solve / discretize)
```

## Measurement layers

| Layer | Group | What it isolates |
| --- | --- | --- |
| Kernel | `SUITE["kernel"]` | Pure arithmetic overhead of the NU reconstruction (Fornberg weights etc.), no symbolic machinery |
| RHS | `SUITE["rhs"]` | One in-place evaluation of the generated ODE right-hand side; the per-step cost unit |
| Solve | `SUITE["solve"]` | Fixed-dt SSPRK33 wall time (adaptivity noise excluded; CFL from `min(dx)`) |
| Discretize | `SUITE["discretize"]` | Symbolic discretization cost (NU stencil expressions are more complex) |

Grid classes: `uniform` (scalar dx, true-uniform path), `uniform_vector` (equispaced nodes
through the NU path - the key overhead-isolation comparison), `stretched`, `perturbed`
(StableRNG seeded).

## Continuous integration

`.github/workflows/benchmark.yml` runs
[the AirspeedVelocity GitHub Action](https://github.com/marketplace/actions/benchmark-pr-with-airspeedvelocity)
on every pull request against `master`. It runs `SUITE` on both the merge-base and the PR
head (freezing the PR's copy of `benchmark/benchmarks.jl` for both revisions) and posts a
runtime/memory comparison table as a PR comment (job summary for fork PRs). Suite sizes in
`benchmarks.jl` are chosen to keep the two-revision job under roughly an hour; superseded
runs on the same PR are cancelled via the workflow's `concurrency` group.

Note that the action does not use `benchmark/Project.toml`; every package the suite
`using`s beyond MethodOfLines and BenchmarkTools must be listed in the workflow's
`extra-pkgs` input.

## Running locally

Compare two revisions with the `benchpkg` CLI:

```bash
julia -e 'using Pkg; Pkg.add("AirspeedVelocity"); Pkg.build("AirspeedVelocity")'
benchpkg MethodOfLines --rev=master,mybranch -s benchmark/benchmarks.jl --exeflags="--threads=1"
benchpkgtable MethodOfLines --rev=master,mybranch
```

Or run the suite directly in the REPL:

```julia
julia --threads=1 --project=benchmark
julia> import Pkg; Pkg.develop(path = "."); Pkg.instantiate()
julia> include("benchmark/benchmarks.jl")
julia> using BenchmarkTools; tune!(SUITE); results = run(SUITE)
```

For heavier scaling sweeps than the CI-sized defaults, build a custom suite with the
keyword arguments of `build_weno_suite` after including `benchmarks.jl`:

```julia
julia> big = build_weno_suite(;
           resolutions = (64, 256, 512),
           interface_resolutions = (41, 81, 161),
           discretize_resolutions = (64, 128),
           interface_discretize_resolutions = (41, 81),
       )
```
