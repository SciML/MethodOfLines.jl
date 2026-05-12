using Printf
using LinearAlgebra

# 1. EXACT ANALYTICAL FUNCTIONS (Method of Manufactured Solutions)
u_exact(x) = sin(2 * pi * x)

# 2. NON-UNIFORM GRID GENERATOR
function generate_nonuniform_grid(N::Int, stretching_factor::Float64=0.1)
    ξ = range(0.0, 1.0, length=N)
    x = @. ξ + stretching_factor * sin(2 * pi * ξ)
    return collect(x) 
end

# 3. WENO ENGINE INTEGRATION
include("weno_weights.jl")
using .WENOWeights

"""
    compute_weno_reconstruction(u, x)

Uses the dynamic WENO weights from PR #538 to reconstruct function values 
at the cell interfaces (x_{i+1/2}) on a strictly non-uniform grid.
"""
function compute_weno_reconstruction(u::Vector{Float64}, x::Vector{Float64})
    N = length(u)
    u_num = zeros(N) 
    u_exact_mid = zeros(N)

    for i in 2:(N-2)
        x_local = x[i-1 : i+1]
        u_local = u[i-1 : i+1]

        x_eval = (x[i] + x[i+1]) / 2.0
        u_exact_mid[i] = u_exact(x_eval)

        d_k = get_nonuniform_ideal_weights(x_local, x_eval)
        beta_k = get_nonuniform_smoothness_indicators(x_local)

        omega_k = get_split_nonlinear_weights(d_k, beta_k)

        u_num[i] = sum(omega_k .* u_local)
    end

    return u_num, u_exact_mid
end

# 4. L2 ERROR NORM AND CONVERGENCE ANALYSIS
function run_convergence_test()
    println("\n============================================================")
    println("   NON-UNIFORM WENO L2 CONVERGENCE TEST (MMS VERIFICATION)  ")
    println("============================================================")
    @printf("%-8s | %-15s | %-15s\n", "N", "L2 Error", "Order of Conv.")
    println("------------------------------------------------------------")

    N_values = [40, 80, 160, 320, 640]
    errors = Float64[]

    for N in N_values
        x = generate_nonuniform_grid(N, 0.1)
        u = u_exact.(x)

        u_num, u_exact_mid = compute_weno_reconstruction(u, x)

        interior_range = 2:(N-2)
        error_sum = 0.0
        for i in interior_range
            error_sum += (u_num[i] - u_exact_mid[i])^2
        end

        L2_error = sqrt(error_sum / length(interior_range))
        push!(errors, L2_error)

        if length(errors) == 1
            @printf("%-8d | %.5e     | %-15s\n", N, L2_error, "-")
        else
            order = log2(errors[end-1] / errors[end])
            @printf("%-8d | %.5e     | %.4f\n", N, L2_error, order)
        end
    end
    println("============================================================\n")
end

run_convergence_test()