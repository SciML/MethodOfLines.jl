# ModelingToolkitNeuralNets networks in a PDE. A scalar field input `NN(u, θ)` maps over
# the field slice on the DAE path and scalarizes on the compiled ODE path. A vector field
# input `NN([u, v], θ)` falls back to pointwise equations.

using MethodOfLines, ModelingToolkit, ModelingToolkitNeuralNets, OrdinaryDiffEq, DomainSets
using SciMLBase
using SciMLBase: successful_retcode
using ModelingToolkit: get_eqs
using Symbolics, SymbolicUtils
using Test
include(joinpath(@__DIR__, "..", "shared", "ode_discretize.jl"))

# Helpers from test/Discretization/equation_discretization.jl.
function isarrayeq(eq)
    isarr(x) = let w = Symbolics.unwrap(x)
        !(w isa AbstractArray) && SymbolicUtils.symtype(w) <: AbstractArray
    end
    return isarr(eq.lhs) || isarr(eq.rhs)
end
isinterioreq(eq) = occursin("Differential(t", string(eq))
narrayeqs_interior(sys) = count(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys))
function treesize(x)
    x = Symbolics.unwrap(x)
    SymbolicUtils.iscall(x) || return 1
    return 1 + sum(treesize, SymbolicUtils.arguments(x); init = 0)
end
interior_treesize(sys) = sum(
    treesize(eq.lhs) + treesize(eq.rhs) for eq in get_eqs(sys) if isinterioreq(eq)
)

@parameters t x
@variables u(..) v(..)
Dt = Differential(t)
Dxx = Differential(x)^2
domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 1.0)]
disc = MOLFiniteDifference([x => 0.05], t)

# Output scaled to O(1) so the term is not negligible. Weights are deterministic.
NN, θ = SymbolicNeuralNetwork(;
    n_input = 1, n_output = 1,
    chain = multi_layer_feed_forward(1, 1; initial_scaling_factor = 1.0),
    nn_p_name = :θ
)

heat_bcs(w) = [w(0, x) ~ sinpi(x), w(t, 0) ~ 0.0, w(t, 1) ~ 0.0]
function heat_system(source; name, ps = [NN, θ])
    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) + source
    return PDESystem(eq, heat_bcs(u), domains, [t, x], [u(t, x)], ps; name)
end

solve_tight(prob) = solve(prob; abstol = 1.0e-10, reltol = 1.0e-10, saveat = 0.02)

@testset "Scalar-input network on the DAE path" begin
    pdesys = heat_system(NN(u(t, x), θ)[1]; name = :nn_heat)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 1
    interior = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)))
    @test occursin("array_map_callable_getindex(", string(interior))
    # Interior expression size is independent of the grid.
    sys_fine, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => 0.0125], t))
    @test interior_treesize(sys_fine) == interior_treesize(sys)

    prob = discretize(pdesys, disc)
    @test prob isa SciMLBase.DAEProblem
    sol = solve_tight(prob)
    @test successful_retcode(sol)

    # Vector input: pointwise reference.
    pdesys_ref = heat_system(NN([u(t, x)], θ)[1]; name = :nn_heat_ref)
    sys_ref, _ = symbolic_discretize(pdesys_ref, disc)
    @test narrayeqs_interior(sys_ref) == 0
    sol_ref = solve_tight(discretize(pdesys_ref, disc))
    @test successful_retcode(sol_ref)
    @test maximum(abs.(sol[u(t, x)] .- sol_ref[u(t, x)])) < 1.0e-8

    # The term is not negligible.
    pdesys_heat = heat_system(0; name = :heat, ps = [])
    sol_heat = solve_tight(discretize(pdesys_heat, disc))
    @test maximum(abs.(sol[u(t, x)] .- sol_heat[u(t, x)])) > 1.0e-3
end

@testset "Compiled ODE path scalarizes the network term" begin
    pdesys = heat_system(NN(u(t, x), θ)[1]; name = :nn_heat_ode)
    prob_ode = ode_discretize(pdesys, disc)
    @test prob_ode isa SciMLBase.ODEProblem
    sol_ode = solve_tight(prob_ode)
    @test successful_retcode(sol_ode)
    sol_dae = solve_tight(discretize(pdesys, disc))
    @test maximum(abs.(sol_ode[u(t, x)] .- sol_dae[u(t, x)])) < 1.0e-6
end

@testset "One network shared by two fields" begin
    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)) + NN(u(t, x), θ)[1],
        Dt(v(t, x)) ~ Dxx(v(t, x)) + NN(v(t, x), θ)[1],
    ]
    @named pdesys = PDESystem(
        eqs, vcat(heat_bcs(u), heat_bcs(v)), domains, [t, x], [u(t, x), v(t, x)], [NN, θ]
    )
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 2
    sol = solve_tight(discretize(pdesys, disc))
    @test successful_retcode(sol)
    @test isapprox(sol[u(t, x)], sol[v(t, x)]; atol = 1.0e-8)
end

@testset "Vector field input falls back and solves" begin
    NN2, θ2 = SymbolicNeuralNetwork(;
        n_input = 2, n_output = 2,
        chain = multi_layer_feed_forward(2, 2; initial_scaling_factor = 1.0),
        nn_name = :NN2, nn_p_name = :θ2
    )
    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)) + NN2([u(t, x), v(t, x)], θ2)[1],
        Dt(v(t, x)) ~ Dxx(v(t, x)) + NN2([u(t, x), v(t, x)], θ2)[2],
    ]
    @named pdesys = PDESystem(
        eqs, vcat(heat_bcs(u), heat_bcs(v)), domains, [t, x], [u(t, x), v(t, x)], [NN2, θ2]
    )
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 0
    sol = solve_tight(discretize(pdesys, disc))
    @test successful_retcode(sol)
end
