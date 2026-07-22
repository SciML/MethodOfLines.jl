# Standalone report driver for the WENO uniform-vs-non-uniform benchmarks.
#
# Usage (from the repository root):
#   julia --threads=1 benchmark/run_benchmarks.jl               # default sweep
#   julia --threads=1 benchmark/run_benchmarks.jl --full        # full scaling sweep
#   julia --threads=1 benchmark/run_benchmarks.jl --smoke       # minimal sanity sweep
#   julia --threads=1 benchmark/run_benchmarks.jl --alloccheck  # + static allocation audit
#
# Outputs (benchmark/results/<timestamp>/):
#   metadata.txt           - versioninfo, thread count, Manifest SHA-256
#   suite.csv              - all BenchmarkTools leaves: min/median time, allocs, memory, GC %
#   rhs_allocations.csv    - in-place RHS allocation audit per problem/grid/N
#   alloccheck.txt         - (--alloccheck only) static AllocCheck audit
#   workprecision.csv, eoc.csv, error_vs_n.png, workprecision.png - smooth-advection sweep
#   front_workprecision.csv, front_eoc.csv, front_crossover.csv,
#   front_error_vs_n.png, front_workprecision.png - moving-front crossover experiment
#   scaling_rhs.png        - RHS time vs N, log-log, per grid class
#   wp_temporal_<kind>.png - DiffEqDevTools tolerance-sweep diagrams

import Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools
using Dates
using InteractiveUtils
using Printf
using SHA
using Statistics
using Plots

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

if "--full" in ARGS
    ENV["MOL_BENCH_MODE"] = "full"
elseif "--smoke" in ARGS
    ENV["MOL_BENCH_MODE"] = "smoke"
end

include(joinpath(@__DIR__, "benchmarks.jl"))
include(joinpath(@__DIR__, "weno", "workprecision.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "results", Dates.format(now(), "yyyy-mm-dd_HHMMSS"))
mkpath(RESULTS_DIR)
outpath(name) = joinpath(RESULTS_DIR, name)

const WP_SWEEP = BENCH_MODE == "smoke" ? (21, 41) :
    BENCH_MODE == "full" ? (41, 81, 161, 321) : (41, 81, 161)

function write_metadata()
    return open(outpath("metadata.txt"), "w") do io
        println(io, "timestamp: ", now())
        println(io, "benchmark mode: ", BENCH_MODE)
        println(io, "julia threads: ", Threads.nthreads())
        manifest = joinpath(@__DIR__, "Manifest.toml")
        if isfile(manifest)
            println(io, "manifest sha256: ", bytes2hex(sha256(read(manifest))))
        end
        println(io, "\n--- versioninfo ---")
        versioninfo(io; verbose = false)
    end
end

function write_csv(name, header, rows)
    return open(outpath(name), "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(string.(row), ","))
        end
    end
end

function run_suite()
    @info "Tuning benchmark suite..."
    tune!(SUITE; verbose = false)
    @info "Running benchmark suite..."
    return run(SUITE; verbose = true)
end

function export_suite(results)
    rows = Vector{Tuple}()
    for (key, trial) in BenchmarkTools.leaves(results)
        tmin = minimum(trial)
        tmed = median(trial)
        gc_pct = 100 * mean(trial.gctimes ./ trial.times)
        push!(
            rows, (
                join(key, "/"), tmin.time, tmed.time,
                tmin.allocs, tmin.memory, @sprintf("%.3f", gc_pct),
            )
        )
    end
    sort!(rows; by = first)
    write_csv(
        "suite.csv",
        ["benchmark", "min_time_ns", "median_time_ns", "allocs", "memory_bytes", "gc_pct"],
        rows,
    )
    return rows
end

# Nonzero RHS allocations compound per step; the audit enforces zero.
function export_rhs_allocations(results)
    rows = Vector{Tuple}()
    for (key, trial) in BenchmarkTools.leaves(results["rhs"])
        tmin = minimum(trial)
        push!(rows, (join(key, "/"), tmin.allocs, tmin.memory, tmin.allocs == 0 ? "PASS" : "NONZERO"))
    end
    sort!(rows; by = first)
    write_csv("rhs_allocations.csv", ["rhs_benchmark", "allocs", "memory_bytes", "status"], rows)
    nonzero = count(r -> r[2] > 0, rows)
    if nonzero > 0
        @warn "$nonzero RHS benchmark(s) allocate; see rhs_allocations.csv"
    else
        @info "All RHS evaluations are allocation-free."
    end
    return rows
end

function plot_rhs_scaling(results)
    plt = plot(;
        xscale = :log10, yscale = :log10,
        xlabel = "N (grid points)", ylabel = "RHS time (ns)",
        title = "WENO RHS evaluation: advection", legend = :topleft,
    )
    adv = results["rhs"]["advection"]
    for kind in sort(collect(keys(adv)))
        pairs_ = BenchmarkTools.leaves(adv[kind])
        ns = Int[]
        ts = Float64[]
        for (key, trial) in pairs_
            push!(ns, parse(Int, split(key[end], "=")[2]))
            push!(ts, minimum(trial).time)
        end
        order = sortperm(ns)
        plot!(plt, ns[order], ts[order]; marker = :circle, label = kind)
    end
    return savefig(plt, outpath("scaling_rhs.png"))
end

function run_workprecision()
    @info "Running spatial work-precision sweep (N = $(WP_SWEEP))..."
    rows = spatial_workprecision(; ns = WP_SWEEP)
    write_csv(
        "workprecision.csv",
        ["grid", "n", "l2_error", "min_solve_time_ns", "allocs", "memory_bytes", "gc_frac"],
        [(r.kind, r.n, r.l2, r.time_ns, r.allocs, r.memory, @sprintf("%.4f", r.gc_frac)) for r in rows],
    )

    eoc = eoc_table(rows)
    write_csv(
        "eoc.csv",
        ["grid", "n_coarse", "n_fine", "observed_order"],
        [(e.kind, e.n_coarse, e.n_fine, @sprintf("%.3f", e.order)) for e in eoc],
    )
    # Formal order 4 in smooth regions; require finest-level EOC > 3 (accuracy-test threshold).
    for kind in unique(e.kind for e in eoc)
        finest = last(sort([e for e in eoc if e.kind == kind]; by = e -> e.n_fine))
        status = finest.order > 3.0 ? "PASS" : "FAIL"
        @info "EOC validation [$kind]: finest-level order = $(round(finest.order; digits = 3)) [$status]"
        status == "FAIL" && @warn "EOC below 3.0 for grid class $kind"
    end

    plt_err = plot(;
        xscale = :log10, yscale = :log10,
        xlabel = "N (grid points)", ylabel = "L2 error (trapezoid-weighted)",
        title = "WENO accuracy: advection, T = 0.5", legend = :bottomleft,
    )
    for kind in unique(r.kind for r in rows)
        sub = sort([r for r in rows if r.kind == kind]; by = r -> r.n)
        plot!(plt_err, [r.n for r in sub], [r.l2 for r in sub]; marker = :circle, label = String(kind))
    end
    savefig(plt_err, outpath("error_vs_n.png"))

    plt_wp = plot(;
        xscale = :log10, yscale = :log10,
        xlabel = "L2 error", ylabel = "solve time (ns, minimum)",
        title = "Work-precision: uniform vs non-uniform WENO", legend = :bottomleft,
        xflip = true,
    )
    for kind in unique(r.kind for r in rows)
        sub = sort([r for r in rows if r.kind == kind]; by = r -> r.n)
        plot!(plt_wp, [r.l2 for r in sub], [r.time_ns for r in sub]; marker = :circle, label = String(kind))
    end
    savefig(plt_wp, outpath("workprecision.png"))

    return rows
end

# Moving-front crossover: error is dominated by spacing inside the front path.

const FRONT_WP_SWEEP = BENCH_MODE == "smoke" ? (81,) :
    BENCH_MODE == "full" ? (81, 161, 321, 641) : (81, 161, 321)

const FRONT_WP_KINDS = (:uniform, :stretched, :front_adapted)

function run_front_workprecision()
    @info "Running moving-front (crossover) work-precision sweep (N = $(FRONT_WP_SWEEP))..."
    sys = front_system()
    rows = spatial_workprecision(
        sys, front_discretization; ns = FRONT_WP_SWEEP, kinds = FRONT_WP_KINDS
    )
    write_csv(
        "front_workprecision.csv",
        ["grid", "n", "l2_error", "min_solve_time_ns", "allocs", "memory_bytes", "gc_frac"],
        [(r.kind, r.n, r.l2, r.time_ns, r.allocs, r.memory, @sprintf("%.4f", r.gc_frac)) for r in rows],
    )

    eoc = eoc_table(rows)
    write_csv(
        "front_eoc.csv",
        ["grid", "n_coarse", "n_fine", "observed_order"],
        [(e.kind, e.n_coarse, e.n_fine, @sprintf("%.3f", e.order)) for e in eoc],
    )

    for n in FRONT_WP_SWEEP
        u = only(r for r in rows if r.kind == :uniform && r.n == n)
        a = only(r for r in rows if r.kind == :front_adapted && r.n == n)
        @info @sprintf(
            "Front equal-N [N=%d]: L2 uniform = %.3e, adapted = %.3e (ratio %.1fx)",
            n, u.l2, a.l2, u.l2 / a.l2
        )
    end

    speedups = equal_error_speedup(rows; base = :uniform, contender = :front_adapted)
    if isempty(speedups)
        @warn "Front crossover: uniform and adapted error ranges do not overlap; extend the N sweep."
    else
        write_csv(
            "front_crossover.csv",
            ["l2_error", "t_uniform_ns", "t_adapted_ns", "speedup"],
            [
                (@sprintf("%.3e", s.l2), round(s.t_base_ns), round(s.t_contender_ns), @sprintf("%.2f", s.speedup))
                    for s in speedups
            ],
        )
        best = maximum(s.speedup for s in speedups)
        won = count(s -> s.speedup > 1, speedups)
        status = won > 0 ? "CROSSOVER CONFIRMED" : "NO CROSSOVER"
        @info @sprintf(
            "Front crossover: adapted faster at %d/%d sampled error levels; peak speedup %.2fx [%s]",
            won, length(speedups), best, status
        )
        for s in speedups
            @info @sprintf(
                "  equal-error L2 = %.2e: uniform %.2f ms vs adapted %.2f ms (%.2fx)",
                s.l2, s.t_base_ns / 1.0e6, s.t_contender_ns / 1.0e6, s.speedup
            )
        end
    end

    plt_err = plot(;
        xscale = :log10, yscale = :log10,
        xlabel = "N (grid points)", ylabel = "L2 error (trapezoid-weighted)",
        title = "Moving tanh front: accuracy vs N", legend = :bottomleft,
    )
    for kind in unique(r.kind for r in rows)
        sub = sort([r for r in rows if r.kind == kind]; by = r -> r.n)
        plot!(plt_err, [r.n for r in sub], [r.l2 for r in sub]; marker = :circle, label = String(kind))
    end
    savefig(plt_err, outpath("front_error_vs_n.png"))

    plt_wp = plot(;
        xscale = :log10, yscale = :log10,
        xlabel = "L2 error", ylabel = "solve time (ns, minimum)",
        title = "Moving-front work-precision: crossover", legend = :bottomleft,
        xflip = true,
    )
    for kind in unique(r.kind for r in rows)
        sub = sort([r for r in rows if r.kind == kind]; by = r -> r.n)
        plot!(plt_wp, [r.l2 for r in sub], [r.time_ns for r in sub]; marker = :circle, label = String(kind))
    end
    savefig(plt_wp, outpath("front_workprecision.png"))

    return rows
end

# Opt-in (GPUCompiler/LLVM load). A failing top-level @testset throws; reaching the write
# implies PASS.
function run_alloccheck_audit()
    "--alloccheck" in ARGS || return nothing
    @info "Running static AllocCheck audit of the WENO kernels..."
    ts = include(joinpath(@__DIR__, "weno", "alloccheck.jl"))
    open(outpath("alloccheck.txt"), "w") do io
        println(io, "testset: ", ts.description)
        println(io, "status: PASS (all static allocation checks passed)")
    end
    return nothing
end

function run_temporal_workprecision()
    BENCH_MODE == "smoke" && return nothing
    @info "Running temporal (tolerance-sweep) work-precision sets..."
    for (kind, wp) in temporal_workprecision()
        plt = plot(wp; title = "Temporal work-precision: $(kind) grid")
        savefig(plt, outpath("wp_temporal_$(kind).png"))
    end
    return nothing
end

write_metadata()
run_alloccheck_audit()
results = run_suite()
export_suite(results)
export_rhs_allocations(results)
plot_rhs_scaling(results)
run_workprecision()
run_front_workprecision()
run_temporal_workprecision()

BenchmarkTools.save(outpath("suite_results.json"), results)
@info "Done. Results written to $(RESULTS_DIR)"
