using SafeTestsets
using SciMLTesting

const FUNCTIONAL_GROUPS = [
    "Components",
    "Complex",
    "Brusselator",
    "Diffusion_NU",
    "Nonlinlap_ADV",
    "Sol_Interface",
    "MOL_Interface2",
    "Diffusion",
    "Integrals",
    "Convection_WENO",
    "Higher_Order",
    "Stationary",
    "Mixed_Derivatives",
    "DAE",
    "Nonlinear_Diffusion",
    "Nonlinear_Diffusion_NU",
    "MOL_Interface1",
    "2D_Diffusion",
    "Burgers",
    "Convection",
    "Convection_NU",
    "Wave_Eq_Staggered",
]

run_tests(;
    core = () -> nothing,
    groups = Dict(
        "Components" => function ()
            @safetestset "MOLFiniteDifference Utils" begin
                include(joinpath(@__DIR__, "Components", "utils_test.jl"))
            end
            @safetestset "Discretization of space and grid types" begin
                include(joinpath(@__DIR__, "Components", "DiscreteSpace.jl"))
            end
            @safetestset "Variable PDE mapping and interior construction" begin
                include(joinpath(@__DIR__, "Components", "interiormap_test.jl"))
            end
            @safetestset "Fornberg" begin
                include(joinpath(@__DIR__, "Components", "MOLfornberg_weights.jl"))
            end
            @safetestset "WENO dispatch" begin
                include(joinpath(@__DIR__, "Components", "weno_dispatch.jl"))
            end
            @safetestset "WENO Non-Uniform Core" begin
                include(joinpath(@__DIR__, "Components", "weno_nonuniform_core.jl"))
            end
            @safetestset "WENO Non-Uniform Boundary" begin
                include(joinpath(@__DIR__, "Components", "weno_nonuniform_boundary.jl"))
            end
            @safetestset "WENO Boundary Integration" begin
                include(joinpath(@__DIR__, "Components", "weno_boundary_integration.jl"))
            end
            @safetestset "ODEFunction" begin
                include(joinpath(@__DIR__, "Components", "ODEFunction_test.jl"))
            end
            @safetestset "MOLFiniteDifference Interface: Staggered constructors" begin
                include(joinpath(@__DIR__, "Components", "staggered_constructors.jl"))
            end
            return @safetestset "Discrete Callbacks" begin
                include(joinpath(@__DIR__, "Components", "callbacks.jl"))
            end
        end,
        "Complex" => joinpath(@__DIR__, "Complex", "schroedinger.jl"),
        "Brusselator" => joinpath(@__DIR__, "Brusselator", "brusselator_eq.jl"),
        "Diffusion_NU" => joinpath(@__DIR__, "Diffusion_NU", "MOL_1D_Linear_Diffusion_NonUniform.jl"),
        "Nonlinlap_ADV" => joinpath(@__DIR__, "Nonlinlap_ADV", "nonlinear_laplacian_advanced.jl"),
        "Sol_Interface" => joinpath(@__DIR__, "Sol_Interface", "solution_interface.jl"),
        "MOL_Interface2" => joinpath(@__DIR__, "MOL_Interface2", "MOLtest2.jl"),
        "Diffusion" => joinpath(@__DIR__, "Diffusion", "MOL_1D_Linear_Diffusion.jl"),
        "Integrals" => joinpath(@__DIR__, "Integrals", "MOL_1D_Integration.jl"),
        "Convection_WENO" => joinpath(@__DIR__, "Convection_WENO", "MOL_1D_Linear_Convection_WENO.jl"),
        "Higher_Order" => joinpath(@__DIR__, "Higher_Order", "MOL_1D_HigherOrder.jl"),
        "Stationary" => joinpath(@__DIR__, "Stationary", "MOL_NonlinearProblem.jl"),
        "Mixed_Derivatives" => joinpath(@__DIR__, "Mixed_Derivatives", "MOL_Mixed_Deriv.jl"),
        "DAE" => joinpath(@__DIR__, "DAE", "MOL_1D_PDAE.jl"),
        "Nonlinear_Diffusion" => joinpath(@__DIR__, "Nonlinear_Diffusion", "MOL_1D_NonLinear_Diffusion.jl"),
        "Nonlinear_Diffusion_NU" => joinpath(@__DIR__, "Nonlinear_Diffusion_NU", "MOL_1D_NonLinear_Diffusion_NonUniform.jl"),
        "MOL_Interface1" => joinpath(@__DIR__, "MOL_Interface1", "MOLtest1.jl"),
        "2D_Diffusion" => joinpath(@__DIR__, "2D_Diffusion", "MOL_2D_Diffusion.jl"),
        "Burgers" => joinpath(@__DIR__, "Burgers", "burgers_eq.jl"),
        "Convection" => joinpath(@__DIR__, "Convection", "MOL_1D_Linear_Convection.jl"),
        "Convection_NU" => joinpath(@__DIR__, "Convection_NU", "MOL_1D_Linear_Convection_NonUniform.jl"),
        "Wave_Eq_Staggered" => joinpath(@__DIR__, "Wave_Eq_Staggered", "wave_eq_staggered.jl"),
    ),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    all = FUNCTIONAL_GROUPS,
)
