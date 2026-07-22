# PDE problems for the WENO benchmarks: smooth periodic advection, inviscid Burgers,
# a moving tanh front (adapted-grid crossover experiment), and a two-domain interface
# pulse. Builders return named tuples consumed by the suite and the report driver.

using ModelingToolkit, DomainSets, MethodOfLines, SciMLBase

# Linear advection: u_t = -u_x, periodic on [0, 2].
function advection_system(; t_end = 0.5)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0, x) ~ sinpi(x),
        u(t, 0.0) ~ u(t, 2.0),
    ]
    domains = [t ∈ Interval(0.0, t_end), x ∈ Interval(0.0, 2.0)]
    @named weno_advection = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    exact(x_, t_) = sinpi(x_ - t_)
    return (;
        pdesys = weno_advection, uvar = u, xvar = x, tvar = t,
        xspan = (0.0, 2.0), t_end, exact,
    )
end

function advection_discretization(sys, kind::Symbol, n::Int)
    spec = grid_spec(kind, sys.xspan..., n)
    return MOLFiniteDifference([sys.xvar => spec], sys.tvar; advection_scheme = WENOScheme())
end

# Inviscid Burgers: shock forms at t ≈ 4/π.
function burgers_system(; t_end = 1.5)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))
    bcs = [
        u(0, x) ~ 1.0 + 0.25 * sinpi(x),
        u(t, 0.0) ~ u(t, 2.0),
    ]
    domains = [t ∈ Interval(0.0, t_end), x ∈ Interval(0.0, 2.0)]
    @named weno_burgers = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    return (;
        pdesys = weno_burgers, uvar = u, xvar = x, tvar = t,
        xspan = (0.0, 2.0), t_end,
    )
end

function burgers_discretization(sys, kind::Symbol, n::Int)
    spec = grid_spec(kind, sys.xspan..., n)
    return MOLFiniteDifference([sys.xvar => spec], sys.tvar; advection_scheme = WENOScheme())
end

# Moving tanh front. The 4th-order error is dominated by the largest spacing inside the
# front path [x0, x0 + t_end]; a band-adapted grid reaches a given L2 error with fewer
# nodes than a uniform grid.
const FRONT_DELTA = 0.02
const FRONT_X0 = 0.55
const FRONT_T_END = 0.2
# Must bracket [x0, x0 + t_end].
const FRONT_BAND = (0.45, 0.85)

function front_system(; delta = FRONT_DELTA, x0 = FRONT_X0, t_end = FRONT_T_END)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    exact(x_, t_) = tanh((x_ - t_ - x0) / delta)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    # Dirichlet at both ends: Neumann outflow yields a singular mass matrix (DAE), forcing
    # implicit solvers. tanh ≡ 1 at x = 2 to machine precision.
    bcs = [
        u(0, x) ~ tanh((x - x0) / delta),
        u(t, 0.0) ~ tanh((-t - x0) / delta),
        u(t, 2.0) ~ tanh((2.0 - t - x0) / delta),
    ]
    domains = [t ∈ Interval(0.0, t_end), x ∈ Interval(0.0, 2.0)]
    @named weno_front = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    return (;
        pdesys = weno_front, uvar = u, xvar = x, tvar = t,
        xspan = (0.0, 2.0), t_end, exact,
    )
end

function front_grid_nodes(kind::Symbol, n::Int)
    kind === :uniform && return uniform_grid(0.0, 2.0, n)
    kind === :stretched && return stretched_grid(0.0, 2.0, n)
    kind === :front_adapted && return front_adapted_grid(0.0, 2.0, n; band = FRONT_BAND)
    return error("front experiment supports :uniform, :stretched, :front_adapted; got $kind")
end

function front_discretization(sys, kind::Symbol, n::Int)
    spec = kind === :uniform ? 2.0 / (n - 1) : front_grid_nodes(kind, n)
    return MOLFiniteDifference([sys.xvar => spec], sys.tvar; advection_scheme = WENOScheme())
end

# Two-domain interface pulse; mirrors test/Convection_WENO/MOL_1D_WENO_NU_Interface.jl.
function interface_system(; t_end = 0.5)
    @parameters t x1 x2
    @variables u1(..) u2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    pulse(x_, t_) = exp(-((x_ - t_) - 0.7)^2 / (2 * 0.1^2))

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(0, x1) ~ pulse(x1, 0.0),
        u2(0, x2) ~ pulse(x2, 0.0),
        u1(t, 0.0) ~ pulse(0.0, t),
        u1(t, 1.0) ~ u2(t, 1.0),
        Dx2(u2(t, 2.0)) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, t_end),
        x1 ∈ Interval(0.0, 1.0),
        x2 ∈ Interval(1.0, 2.0),
    ]
    @named weno_interface = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)]
    )

    return (;
        pdesys = weno_interface, u1var = u1, u2var = u2,
        x1var = x1, x2var = x2, tvar = t, t_end, exact = pulse,
    )
end

# Seam requires vector grids; interval counts mismatched (n : 3n/2) to exercise the seam.
function interface_discretization(sys, kind::Symbol, n::Int)
    n2 = round(Int, 3 * n / 2)
    if kind === :uniform_vector
        g1 = uniform_grid(0.0, 1.0, n)
        g2 = uniform_grid(1.0, 2.0, n2)
    elseif kind === :stretched
        g1 = stretched_grid(0.0, 1.0, n; amp = 0.03)
        g2 = stretched_grid(1.0, 2.0, n2; amp = 0.04)
    else
        error("interface benchmark supports :uniform_vector and :stretched, got $kind")
    end
    @assert all(diff(g1) .> 0) && all(diff(g2) .> 0)
    return MOLFiniteDifference(
        [sys.x1var => g1, sys.x2var => g2], sys.tvar; advection_scheme = WENOScheme()
    )
end

interface_min_spacing(kind::Symbol, n::Int) = min(
    kind === :uniform_vector ? 1.0 / (n - 1) : minimum(diff(stretched_grid(0.0, 1.0, n; amp = 0.03))),
    kind === :uniform_vector ? 1.0 / (round(Int, 3 * n / 2) - 1) :
        minimum(diff(stretched_grid(1.0, 2.0, round(Int, 3 * n / 2); amp = 0.04))),
)
