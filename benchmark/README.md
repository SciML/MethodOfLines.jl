# MethodOfLines.jl WENO Benchmarks

Benchmark infrastructure comparing the non-uniform WENO-5 implementation against its
uniform counterpart, following the PkgBenchmark.jl / AirspeedVelocity.jl convention
(`benchmark/benchmarks.jl` defines `const SUITE :: BenchmarkGroup`).

## Layout

```
benchmark/
  Project.toml          independent environment (BenchmarkTools, DiffEqDevTools, Plots, ...)
  benchmarks.jl         SUITE definition (PkgBenchmark / AirspeedVelocity entry point)
  weno/
    grids.jl            grid classes + trapezoid-weighted L2 norm
    problems.jl         PDE definitions (advection, Burgers, two-domain interface)
    suite.jl            BenchmarkGroup hierarchy (kernel / rhs / solve / discretize)
    workprecision.jl    equal-error comparison and EOC validation
    alloccheck.jl       static allocation audit of the WENO kernels (local-only)
  run_benchmarks.jl     standalone report driver (CSV + plots + metadata)
  results/              driver outputs (gitignored)
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
(StableRNG seeded). Errors use the trapezoid-weighted discrete L2 norm, since plain RMS is
inconsistent on non-uniform grids.

## Moving-front crossover experiment

The smooth advection sweep measures the *cost* of the NU capability; the moving tanh-front
experiment (`front_system` in `problems.jl`) measures its *payoff*. A sharp front
`u = tanh((x - t - x0)/δ)` with `δ = 0.02` is advected across `[0, 2]`, and a
density-equidistributed grid (`front_adapted_grid`) concentrates ~2/3 of the nodes on the
front's path. The report driver emits `front_workprecision.png` (the crossover diagram),
`front_error_vs_n.png`, `front_eoc.csv`, and `front_crossover.csv` with equal-error
`t_uniform / t_adapted` speedups. Expected qualitative picture: uniform grids plateau at
O(1) error until `h ≪ δ`, while the adapted grid is orders of magnitude more accurate at
equal N and ~2-2.6x faster at equal error.

## Running

Quick suite run in the REPL:

```julia
julia --threads=1 --project=benchmark
julia> import Pkg; Pkg.develop(path="."); Pkg.instantiate()
julia> include("benchmark/benchmarks.jl")
julia> using BenchmarkTools; tune!(SUITE); results = run(SUITE)
```

Full report (CSV tables, scaling / error / work-precision plots, reproducibility metadata):

```powershell
julia --threads=1 benchmark/run_benchmarks.jl               # default sweep
julia --threads=1 benchmark/run_benchmarks.jl --full        # full scaling sweep
julia --threads=1 benchmark/run_benchmarks.jl --smoke       # minimal sanity sweep
julia --threads=1 benchmark/run_benchmarks.jl --alloccheck  # + static allocation audit
```

`ENV["MOL_BENCH_MODE"] ∈ ("smoke", "default", "full")` controls the resolution sweep when
loading `benchmarks.jl` directly.

## Comparing two revisions

With [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl):

```bash
benchpkg MethodOfLines --rev=master,mybranch --exeflags="--threads=1"
benchpkgtable MethodOfLines --rev=master,mybranch
```

With [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl):

```julia
using PkgBenchmark
judge("MethodOfLines", "master")
```

## Allocation guarantees

Static allocation-freedom of the WENO kernels is proven (not sampled) by AllocCheck in
`benchmark/weno/alloccheck.jl`. This audit is deliberately local-only: AllocCheck pulls the
GPUCompiler/LLVM toolchain, which is too heavy and Julia-version-sensitive for the package's
CI test matrix. Run it standalone or via the driver flag:

```powershell
julia --threads=1 benchmark/weno/alloccheck.jl
julia --threads=1 benchmark/run_benchmarks.jl --alloccheck
```

Lightweight `@allocated`-based allocation checks remain in CI as part of the `Components`
test group (`test/Components/weno_nonuniform_core.jl`, `test/Components/weno_dispatch.jl`).
The report driver additionally audits the in-place RHS for zero allocations per call
(`rhs_allocations.csv`).
