# ModelingToolkitNeuralNets networks in a PDE. `NN(u, θ)` and `NN([u, v], θ)` stay in
# array form on the DAE path and scalarize on the compiled ODE path. The extension
# evaluates a network on the whole slice in one call; the same wrapper without the
# network metadata is the point-by-point reference.

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

nn_wrapper = ModelingToolkit.getdefault(NN)
@parameters (NNpp::typeof(nn_wrapper))(..)[1:1] = nn_wrapper [tunable = false]
NN2, θ2 = SymbolicNeuralNetwork(;
    n_input = 2, n_output = 2,
    chain = multi_layer_feed_forward(2, 2; initial_scaling_factor = 1.0),
    nn_name = :NN2, nn_p_name = :θ2
)
nn2_wrapper = ModelingToolkit.getdefault(NN2)
@parameters (NN2pp::typeof(nn2_wrapper))(..)[1:2] = nn2_wrapper [tunable = false]

heat_bcs(w) = [w(0, x) ~ sinpi(x), w(t, 0) ~ 0.0, w(t, 1) ~ 0.0]
function heat_system(source; name, ps = [NN, θ])
    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) + source
    return PDESystem(eq, heat_bcs(u), domains, [t, x], [u(t, x)], ps; name)
end

solve_tight(prob) = solve(prob; abstol = 1.0e-10, reltol = 1.0e-10, saveat = 0.02)

@testset "Scalar-input network on the DAE path" begin
    @test Base.get_extension(MethodOfLines, :MethodOfLinesModelingToolkitNeuralNetsExt) !==
        nothing
    pdesys = heat_system(NN(u(t, x), θ)[1]; name = :nn_heat)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 1
    interior = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)))
    @test occursin("array_batch_callable_getindex(", string(interior))
    # Interior expression size is independent of the grid.
    sys_fine, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => 0.0125], t))
    @test interior_treesize(sys_fine) == interior_treesize(sys)

    prob = discretize(pdesys, disc)
    @test prob isa SciMLBase.DAEProblem
    sol = solve_tight(prob)
    @test successful_retcode(sol)

    # Point-by-point reference: the same network without the metadata.
    pdesys_pp = heat_system(NNpp(u(t, x), θ)[1]; name = :nn_heat_pp, ps = [NNpp, θ])
    sys_pp, _ = symbolic_discretize(pdesys_pp, disc)
    interior_pp = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_pp)))
    @test occursin("array_map_callable_getindex(", string(interior_pp))
    sol_pp = solve_tight(discretize(pdesys_pp, disc))
    @test successful_retcode(sol_pp)
    @test maximum(abs.(sol[u(t, x)] .- sol_pp[u(t, x)])) < 1.0e-8

    # A one-element vector input is a stack of one slice.
    pdesys_vec = heat_system(NN([u(t, x)], θ)[1]; name = :nn_heat_vec)
    sys_vec, _ = symbolic_discretize(pdesys_vec, disc)
    @test narrayeqs_interior(sys_vec) == 1
    interior_vec = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_vec)))
    @test occursin("array_batch_callable_stacked_getindex(", string(interior_vec))
    sol_vec = solve_tight(discretize(pdesys_vec, disc))
    @test successful_retcode(sol_vec)
    @test maximum(abs.(sol[u(t, x)] .- sol_vec[u(t, x)])) < 1.0e-8

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

@testset "Vector field input is stacked and batched" begin
    function two_field(net, ps, name)
        uv = net([u(t, x), v(t, x)], ps[end])
        eqs = [Dt(u(t, x)) ~ Dxx(u(t, x)) + uv[1], Dt(v(t, x)) ~ Dxx(v(t, x)) + uv[2]]
        return PDESystem(
            eqs, vcat(heat_bcs(u), heat_bcs(v)), domains, [t, x], [u(t, x), v(t, x)], ps;
            name
        )
    end
    pdesys = two_field(NN2, [NN2, θ2], :nn_two)
    sys, _ = symbolic_discretize(pdesys, disc)
    interior = filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys))
    @test length(interior) == 2
    @test all(eq -> occursin("array_batch_callable_stacked_getindex(", string(eq)), interior)
    sol = solve_tight(discretize(pdesys, disc))
    @test successful_retcode(sol)

    pdesys_pp = two_field(NN2pp, [NN2pp, θ2], :nn_two_pp)
    sys_pp, _ = symbolic_discretize(pdesys_pp, disc)
    interior_pp = filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_pp))
    @test all(eq -> occursin("array_map_callable_stacked_getindex(", string(eq)), interior_pp)
    sol_pp = solve_tight(discretize(pdesys_pp, disc))
    @test successful_retcode(sol_pp)
    @test maximum(abs.(sol[u(t, x)] .- sol_pp[u(t, x)])) < 1.0e-8
    @test maximum(abs.(sol[v(t, x)] .- sol_pp[v(t, x)])) < 1.0e-8

    sol_ode = solve_tight(ode_discretize(pdesys, disc))
    @test successful_retcode(sol_ode)
    @test maximum(abs.(sol_ode[u(t, x)] .- sol[u(t, x)])) < 1.0e-6
end

@testset "Brusselator with a network term" begin
    α = 0.1
    uv = NN2([u(t, x), v(t, x)], θ2)
    eqs = [
        Dt(u(t, x)) ~ 1 + u(t, x)^2 * v(t, x) - 4.4 * u(t, x) + α * Dxx(u(t, x)) + uv[1],
        Dt(v(t, x)) ~ 3.4 * u(t, x) - u(t, x)^2 * v(t, x) + α * Dxx(v(t, x)) + uv[2],
    ]
    bcs = [
        u(0, x) ~ 22 * (x * (1 - x))^(3 / 2), v(0, x) ~ 27 * (x * (1 - x))^(3 / 2),
        u(t, 0) ~ u(t, 1), v(t, 0) ~ v(t, 1),
    ]
    bruss_domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
    @named bruss = PDESystem(eqs, bcs, bruss_domains, [t, x], [u(t, x), v(t, x)], [NN2, θ2])
    # The periodic seam points stay pointwise; their number does not grow with the grid.
    nscalar_interior(sys) = count(!isarrayeq, filter(isinterioreq, get_eqs(sys)))
    sys, _ = symbolic_discretize(bruss, disc)
    @test narrayeqs_interior(sys) == 2
    sys_fine, _ = symbolic_discretize(bruss, MOLFiniteDifference([x => 0.025], t))
    @test nscalar_interior(sys_fine) == nscalar_interior(sys)
    sol = solve(discretize(bruss, disc); saveat = 0.1)
    @test successful_retcode(sol)
end

@testset "Stacked network input on a 2D grid" begin
    @parameters y
    Dyy = Differential(y)^2
    uv = NN2([u(t, x, y), v(t, x, y)], θ2)
    eqs = [
        Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + uv[1],
        Dt(v(t, x, y)) ~ Dxx(v(t, x, y)) + Dyy(v(t, x, y)) + uv[2],
    ]
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y), v(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0, u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
        v(t, 0, y) ~ 0.0, v(t, 1, y) ~ 0.0, v(t, x, 0) ~ 0.0, v(t, x, 1) ~ 0.0,
    ]
    domains2 = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(
        eqs, bcs, domains2, [t, x, y], [u(t, x, y), v(t, x, y)], [NN2, θ2]
    )
    disc2 = MOLFiniteDifference([x => 0.2, y => 0.2], t)
    sys, _ = symbolic_discretize(pdesys, disc2)
    @test narrayeqs_interior(sys) == 2
    sol = solve(discretize(pdesys, disc2); saveat = 0.05)
    @test successful_retcode(sol)
end
