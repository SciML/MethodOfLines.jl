# Wrapping a solution whose states live on the GPU must not scalar-index them, and must
# give exactly the wrapper the same states produce on the host.
using CUDA, MethodOfLines, ModelingToolkit, ModelingToolkitBase, OrdinaryDiffEq,
    NonlinearSolve, DomainSets, SciMLBase, Symbolics, SymbolicUtils, Test
using ModelingToolkit: Differential

# More than 16 points per field, so no extraction goes through a small-array scalar path.
const N = 40

host_states(sol) = SciMLBase.build_solution(
    sol.prob, sol.alg, sol.t, Array.(sol.u); retcode = sol.retcode
)
device_states(sol) = SciMLBase.build_solution(
    sol.prob, sol.alg, sol.t, CuArray.(sol.u); retcode = sol.retcode
)

function test_same_wrapper(gpu_wrapped, cpu_wrapped)
    @test keys(gpu_wrapped.u) == keys(cpu_wrapped.u)
    for k in keys(cpu_wrapped.u)
        @test gpu_wrapped.u[k] isa Array
        @test gpu_wrapped.u[k] == cpu_wrapped.u[k]
    end
    return maximum(maximum(abs.(gpu_wrapped.u[k] .- cpu_wrapped.u[k])) for k in keys(cpu_wrapped.u))
end

if !CUDA.functional()
    @info "CUDA is not functional; skipping GPU solution wrapping tests."
    @test_skip false
else
    # Every GPU array access below that is not explicitly `@allowscalar` must be bulk.
    CUDA.allowscalar(false)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    @named heat = PDESystem(
        Dt(u(t, x)) ~ Dxx(u(t, x)),
        [u(0, x) ~ sin(pi * x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]
    )
    disc = MOLFiniteDifference([x => 1 / N], t)

    @testset "ODEProblem solved on the GPU" begin
        sys, tspan = symbolic_discretize(heat, disc)
        cpu_prob = ODEProblem(mtkcompile(sys), nothing, tspan)
        metadata = solve(cpu_prob, Tsit5(); saveat = 0.01).disc_data
        # `mtkcompile` moves the Dirichlet boundary points into observed equations, so
        # this covers both the unknown and the observed extraction paths.
        @test length(cpu_prob.u0) == N - 1
        # FullSpecialize keeps the function wrapper from converting the states back to
        # `Vector`. The generated RHS writes element by element, so the solve itself
        # needs scalar indexing; only the wrapping is checked without it.
        f = ODEFunction{true, SciMLBase.FullSpecialize}(
            cpu_prob.f.f; sys = cpu_prob.f.sys, observed = cpu_prob.f.observed
        )
        gpu_prob = ODEProblem(f, CuArray(cpu_prob.u0), cpu_prob.tspan, cpu_prob.p)
        raw = CUDA.@allowscalar solve(
            gpu_prob, Tsit5(); saveat = 0.01, wrap = Val(false),
            initializealg = SciMLBase.NoInit()
        )
        @test SciMLBase.successful_retcode(raw)
        @test raw.u[end] isa CuArray
        gpu_wrapped = SciMLBase.wrap_sol(raw, metadata)
        cpu_wrapped = SciMLBase.wrap_sol(host_states(raw), metadata)
        maxdiff = test_same_wrapper(gpu_wrapped, cpu_wrapped)
        @info "ODE GPU-solved wrap: max |GPU wrap - CPU wrap| = $maxdiff"
        @test size(gpu_wrapped[u(t, x)]) == (length(raw.t), N + 1)
    end

    @testset "DAEProblem with device states" begin
        prob = discretize(heat, disc)
        @test prob isa DAEProblem
        cpu_wrapped = solve(prob; saveat = 0.01)
        gpu_wrapped = SciMLBase.wrap_sol(
            device_states(cpu_wrapped.original_sol), cpu_wrapped.disc_data
        )
        maxdiff = test_same_wrapper(gpu_wrapped, cpu_wrapped)
        @info "DAE device-state wrap: max |GPU wrap - CPU wrap| = $maxdiff"
    end

    @testset "NonlinearProblem with device states" begin
        @parameters y
        @variables v(..)
        Dyy = Differential(y)^2
        @named poisson = PDESystem(
            Dyy(v(y)) ~ -1.0, [v(0) ~ 0.0, v(1) ~ 0.0], [y ∈ Interval(0.0, 1.0)],
            [y], [v(y)]
        )
        prob = discretize(poisson, MOLFiniteDifference([y => 1 / N], nothing))
        cpu_wrapped = solve(prob, NewtonRaphson())
        sol = cpu_wrapped.original_sol
        device_sol = SciMLBase.build_solution(
            sol.prob, sol.alg, CuArray(sol.u), sol.resid; retcode = sol.retcode
        )
        gpu_wrapped = SciMLBase.wrap_sol(device_sol, cpu_wrapped.disc_data)
        maxdiff = test_same_wrapper(gpu_wrapped, cpu_wrapped)
        @info "Nonlinear device-state wrap: max |GPU wrap - CPU wrap| = $maxdiff"
    end

    @testset "Array observed equations with device states" begin
        # A reduced model whose field is an observed array `w = basis * reduced`, which
        # the wrapper evaluates in one call per time point instead of per element.
        @variables w(..) reduced(t)[1:2]
        @named decay = PDESystem(
            Dt(w(t, x)) ~ -w(t, x), [w(0, x) ~ 1 + x],
            [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 1.0)], [t, x], [w(t, x)]
        )
        source, _ = symbolic_discretize(decay, disc)
        metadata = SymbolicUtils.getmetadata(
            source, ModelingToolkitBase.ProblemTypeCtx, nothing
        )
        entry = Symbolics.unwrap(first(metadata.discretespace.discvars[w(t, x)]))
        field = first(SymbolicUtils.arguments(entry))
        basis = hcat(ones(N + 1), range(0.0, 1.0; length = N + 1))
        @named rom = System(
            [Dt(reduced) ~ -reduced], t,
            Symbolics.unwrap.(Symbolics.scalarize(reduced)), [];
            observed = [field ~ basis * reduced]
        )
        observed = function (sym, state, p, time)
            isequal(Symbolics.unwrap(sym), field) || error("unexpected request: $sym")
            return basis * state
        end
        f = ODEFunction{true}(
            (du, u, p, t) -> (du .= -u); sys = ModelingToolkitBase.complete(rom), observed
        )
        prob = ODEProblem(f, ones(2), (0.0, 0.2))
        times = [0.0, 0.1, 0.2]
        states = [fill(exp(-time), 2) for time in times]
        cpu_wrapped = SciMLBase.wrap_sol(
            SciMLBase.build_solution(prob, nothing, times, states), metadata
        )
        gpu_wrapped = SciMLBase.wrap_sol(
            SciMLBase.build_solution(prob, nothing, times, CuArray.(states)), metadata
        )
        maxdiff = test_same_wrapper(gpu_wrapped, cpu_wrapped)
        @info "Array-observed device-state wrap: max |GPU wrap - CPU wrap| = $maxdiff"
    end
end
