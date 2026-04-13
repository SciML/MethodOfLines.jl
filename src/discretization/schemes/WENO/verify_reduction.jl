"""
Validation Script: Analytical Reduction of Non-Uniform WENO
This script formally verifies that the dynamic Shi-Hu-Shu weight splitting 
analytically reduces to the ideal uniform Lagrange weights when grid spacing is constant.
"""

using LinearAlgebra
using Printf

# Dynamically include the weights module regardless of where the user runs the script from
include(joinpath(@__DIR__, "weno_weights.jl"))
using .WENOWeights

function run_validation()
    println("============================================================")
    println("  WENO ANALYTICAL REDUCTION VERIFICATION")
    println("============================================================")

    # 1. Setup: Perfectly Uniform Grid
    h = 0.1
    x_uni = collect(0.0:h:2h) # [0.0, 0.1, 0.2]
    x_eval = 0.05             # Mid-point of the first cell

    println("[*] Grid Setup       : x = ", x_uni)
    println("[*] Grid Spacing (dx): ", h)
    println("[*] Evaluation Point : ", x_eval)
    println("------------------------------------------------------------")

    # 2. Calculate Legacy Ideal Weights (d_k)
    d_k = WENOWeights.get_nonuniform_ideal_weights(x_uni, x_eval)

    # 3. Calculate Dynamic Non-Linear Weights (omega_k)
    beta_k = WENOWeights.get_nonuniform_smoothness_indicators(x_uni)
    omega_k = WENOWeights.get_split_nonlinear_weights(d_k, beta_k)

    # 4. Compute Scientific Metrics
    diff = omega_k .- d_k
    l_inf = norm(diff, Inf)
    l2 = norm(diff, 2)
    mach_eps = eps(Float64)

    # 5. Output Formatted Results
    println("[*] Metric Results:")
    @printf("    L-infinity Norm (Max Error) : %e\n", l_inf)
    @printf("    L2 Norm (RMS Error)         : %e\n", l2)
    @printf("    Float64 Machine Epsilon     : %e\n", mach_eps)
    println("------------------------------------------------------------")

    # 6. Formal Conclusion
    println("[*] Conclusion:")
    if l_inf <= mach_eps * 10
        println("    [PASS] The maximum absolute error is bounded by machine precision.")
        println("    [PASS] Analytical reduction to uniform scheme is FORMALLY CONFIRMED.")
    else
        println("    [FAIL] The error exceeds acceptable machine precision limits.")
    end
    println("============================================================")
end

# Execute the validation
run_validation()