using SafeTestsets
import Base.Threads.@spawn

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")
const is_TRAVIS = haskey(ENV, "TRAVIS")

# Start Test Script

@time begin

    if GROUP == "All" || GROUP == "Components"
        #@time @safetestset "Test for regression against original code" begin include("regression_test.jl") end
        @time @safetestset "MOLFiniteDifference Utils" begin
            include("utils_test.jl")
        end
        @time @safetestset "Discretization of space and grid types" begin
            include("components/DiscreteSpace.jl")
        end
        @time @safetestset "Variable PDE mapping and interior construction" begin
            include("components/interiormap_test.jl")
        end
        @time @safetestset "Fornberg" begin
            include("components/MOLfornberg_weights.jl")
        end
        #@time @safetestset "Finite Difference Schemes" begin include("components/finite_diff_schemes.jl") end
    end

    if GROUP == "All" || GROUP == "Interface"
        @time @safetestset "MOLFiniteDifference Interface" begin
            include("pde_systems/MOLtest.jl")
        end
    end

    if GROUP == "All" || GROUP == "Diffusion"
        @time @safetestset "MOLFiniteDifference Interface: 1D Linear Diffusion" begin
            include("pde_systems/MOL_1D_Linear_Diffusion.jl")
        end
    end

    if GROUP == "All" || GROUP == "Diffusion_NU"
        @time @safetestset "MOLFiniteDifference Interface: 1D Linear Diffusion, Non-Uniform" begin
            include("pde_systems/MOL_1D_Linear_Diffusion_NonUniform.jl")
        end
    end

    if GROUP == "All" || GROUP == "Nonlinear_Diffusion"
        @time @safetestset "MOLFiniteDifference Interface: 1D Non-Linear Diffusion" begin
            include("pde_systems/MOL_1D_NonLinear_Diffusion.jl")
        end
    end

    if GROUP == "All" || GROUP == "Nonlinear_Diffusion_NU"
        @time @safetestset "MOLFiniteDifference Interface: 1D Non-Linear Diffusion, Non-Uniform" begin
            include("pde_systems/MOL_1D_NonLinear_Diffusion_NonUniform.jl")
        end
    end

    if GROUP == "All" || GROUP == "2D_Diffusion"
        @time @safetestset "MOLFiniteDifference Interface: 2D Diffusion" begin
            include("pde_systems/MOL_2D_Diffusion.jl")
        end
    end

    if GROUP == "All" || GROUP == "Convection"
        @time @safetestset "MOLFiniteDifference Interface: Linear Convection" begin
            include("pde_systems/MOL_1D_Linear_Convection.jl")
        end
    end

    if GROUP == "All" || GROUP == "Higher_Order"
        @time @safetestset "MOLFiniteDifference Interface: 1D HigherOrder" begin
            include("pde_systems/MOL_1D_HigherOrder.jl")
        end
    end

    if GROUP == "All" || GROUP == "Stationary"
        @time @safetestset "MOLFiniteDifference Interface: Stationary Nonlinear Problems" begin
            include("pde_systems/MOL_NonlinearProblem.jl")
        end
    end

    if GROUP == "All" || GROUP == "DAE"
        @time @safetestset "MOLFiniteDifference Interface: 1D Partial DAE" begin
            include("pde_systems/MOL_1D_PDAE.jl")
        end
    end

    if GROUP == "All" || GROUP == "Burgers"
        @time @safetestset "MOLFiniteDifference Interface: 2D Burger's Equation" begin
            include("pde_systems/burgers_eq.jl")
        end
    end

    if GROUP == "All" || GROUP == "Brusselator"
        @time @safetestset "MOLFiniteDifference Interface: 2D Brusselator Equation" begin
            include("pde_systems/brusselator_eq.jl")
        end
    end
end
