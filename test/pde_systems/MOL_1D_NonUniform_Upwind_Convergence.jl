using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, Test, Printf

@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)

# ─── Strongly non-uniform logarithmic grid ───────────────────────────────────
# Dense at x = 0, sparse at x = 1. Stretching ratio Δx_max/Δx_min ≈ exp(α).
function log_grid_strong(N::Int, ::Type{T} = Float64; α = T(5.5)) where {T}
    ts = range(zero(T), one(T); length = N)
    return collect(@. (exp(α * ts) - one(T)) / (exp(α) - one(T)))
end

# ─── Manufactured solution: smooth half-period sine wave ─────────────────────
# u(t, x) = sin(π(x − t) / 2). Bounded by 1, |u_xx| ≤ (π/2)² ≈ 2.47, so the
# leading-order truncation term is finite under non-uniform stretching and
# the asymptotic regime is reached at modest N.
const C_ADV   = 1.0
const T_FINAL = 0.1
u_exact(t, x) = sin(π * (x - C_ADV * t) / 2)

eq      = Dt(u(t, x)) + C_ADV * Dx(u(t, x)) ~ 0
bcs(domain_x_lo, domain_x_hi) = [
    u(0, x) ~ sin(π * x / 2),
    u(t, domain_x_lo) ~ sin(-π * C_ADV * t / 2),
    u(t, domain_x_hi) ~ sin(π * (domain_x_hi - C_ADV * t) / 2),
]
domains = [t ∈ Interval(0.0, T_FINAL), x ∈ Interval(0.0, 1.0)]

function l∞_error(N::Int)
    xg = log_grid_strong(N)
    @named pdesys = PDESystem(eq, bcs(0.0, 1.0), domains, [t, x], [u(t, x)])
    disc = MOLFiniteDifference([x => xg], t; advection_scheme = UpwindScheme(1))
    prob = discretize(pdesys, disc)
    sol  = solve(prob, Rodas5P(); reltol = 1e-10, abstol = 1e-12,
                 saveat = [T_FINAL])
    @assert SciMLBase.successful_retcode(sol.retcode) "solve failed at N=$N"
    x_sol = sol[x]
    u_num = sol[u(t, x)][end, :]
    u_ref = u_exact.(T_FINAL, x_sol)
    return maximum(abs.(u_num .- u_ref))
end

@testset "Pillar 1 — Convergence on strongly non-uniform grid" begin

    @testset "Grid stretching ratio audit" begin
        # Each refinement level must place the grid in the targeted band
        # 1 : 100 .. 1 : 500. This proves the non-uniform dispatch is exercised
        # at every N while remaining inside the asymptotic-regime envelope.
        for N in (80, 160, 320)
            xg    = log_grid_strong(N)
            dxs   = diff(xg)
            ratio = maximum(dxs) / minimum(dxs)
            @test 100.0 < ratio < 500.0
            @printf "  N = %3d   Δx_min = %.3e   Δx_max = %.3e   ratio = %.1f\n" N minimum(dxs) maximum(dxs) ratio
        end
    end

    @testset "L∞ convergence O(Δx)" begin
        Ns      = (80, 160, 320)
        errors  = [l∞_error(N) for N in Ns]
        ratios  = [errors[i - 1] / errors[i] for i in 2:length(errors)]
        orders  = log2.(ratios)

        println()
        @printf "  %-6s  %-15s  %-12s  %-10s\n" "N" "L∞ error" "E(N)/E(2N)" "log₂"
        println(repeat('-', 50))
        @printf "  %-6d  %-15.6e  %-12s  %-10s\n" Ns[1] errors[1] "—" "—"
        for i in 2:length(Ns)
            @printf "  %-6d  %-15.6e  %-12.4f  %-10.4f\n" Ns[i] errors[i] ratios[i - 1] orders[i - 1]
        end
        println()

        # Strict assertion: log₂(E(N)/E(2N)) ∈ [0.90, 1.05] for both consecutive
        # refinements. This is the canonical signature of first-order spatial
        # accuracy under doubling-N refinement, evaluated fully inside the
        # asymptotic regime where the stretching term has decayed.
        for ord in orders
            @test 0.90 ≤ ord ≤ 1.05
        end

        # Defensive sanity: every error level must be a finite real number.
        @test all(isfinite, errors)
    end
end
