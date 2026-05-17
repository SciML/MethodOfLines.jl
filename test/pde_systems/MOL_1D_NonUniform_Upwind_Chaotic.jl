using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, Test

@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)

# Velocity field with five interior sign reversals (zeros at multiples of 0.1).
v_field(x) = sin(10π * x)

# Logarithmic non-uniform grid — combines a non-uniform geometry with a
# rapidly oscillating advection coefficient, the worst-case sympathetic
# stress for the Fornberg dispatch.
function log_grid(N::Int; α = 2.0)
    ts = range(0.0, 1.0; length = N)
    return collect(@. (exp(α * ts) - 1) / (exp(α) - 1))
end

# Initial profile: smooth Gaussian centred in the domain. Bounded by 1.
u0_profile(x) = exp(-((x - 0.5) / 0.1)^2)

# v(0) = v(1) = 0, so the spatial boundary is a degenerate inflow/outflow
# where the PDE reduces to ∂_t u = 0 — the IC values are stationary BCs.
const U_BC_LEFT  = u0_profile(0.0)   # ≈ exp(−25)
const U_BC_RIGHT = u0_profile(1.0)   # ≈ exp(−25)

@testset "Pillar 3 — Chaotic variable wind v(x) = sin(10πx)" begin
    eq   = Dt(u(t, x)) + v_field(x) * Dx(u(t, x)) ~ 0
    bcs  = [
        u(0, x) ~ u0_profile(x),
        u(t, 0) ~ U_BC_LEFT,
        u(t, 1) ~ U_BC_RIGHT,
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    xg   = log_grid(40)
    disc = MOLFiniteDifference([x => xg], t; advection_scheme = UpwindScheme(1))

    @testset "discretize() resolves dispatch at every node" begin
        # The discretize call must traverse generate_winding_rules, which in
        # turn invokes upwind_difference(expr, ...) at every interior point.
        # If either Fornberg branch were unreachable or malformed, this would
        # throw before any ODEProblem could be returned.
        prob = discretize(pdesys, disc)
        @test prob isa ODEProblem
    end

    @testset "solve() completes and produces only finite values" begin
        prob = discretize(pdesys, disc)
        # Rodas5P: the rapid sign flipping makes the linearised operator mildly
        # stiff under non-uniform spacing. An L-stable solver isolates the
        # spatial dispatch from any time-integration artefacts.
        sol  = solve(prob, Rodas5P(); reltol = 1e-8, abstol = 1e-10,
                     saveat = [0.05, 0.1])

        @test SciMLBase.successful_retcode(sol.retcode)

        u_arr = sol[u(t, x)]
        @test all(isfinite, u_arr)
        @test !any(isnan, u_arr)
        @test !any(isinf, u_arr)
    end

    @testset "Direct rhs evaluation at t = 0 is finite" begin
        prob = discretize(pdesys, disc)
        f    = prob.f.f
        u0   = copy(prob.u0)
        du   = similar(u0)
        f(du, u0, prob.p, 0.0)

        @test all(isfinite, du)
        @test eltype(du) === Float64
        @test !any(isnan, du)
        @test !any(isinf, du)
    end
end
