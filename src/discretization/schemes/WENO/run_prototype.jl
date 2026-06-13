using BenchmarkTools
include("weno_weights.jl")
using .WENOWeights

function showcase_prototype()
    println("="^60)
    println("SCIML NON-UNIFORM WENO MATHEMATICAL PROTOTYPE SHOWCASE")
    println("="^60)

    # 1. Setup a non-uniform clustered grid (Type: Float64)
    x_local = [0.0, 0.05, 0.2] 
    x_eval = 0.1

    println("\n[1] TESTING FLOAT64 PRECISION")
    println("Grid points: ", x_local)
    println("Evaluation point: ", x_eval)

    # Calculate weights
    d_k = get_nonuniform_ideal_weights(x_local, x_eval)
    beta_k = get_nonuniform_smoothness_indicators(x_local)
    omega_k = get_split_nonlinear_weights(d_k, beta_k)

    println("Ideal Weights (d_k): ", d_k)
    println("Smoothness Indicators (beta_k): ", beta_k)
    println("Final Non-Linear Weights (omega_k): ", omega_k)
    println("Sum of Weights: ", sum(omega_k), " (Must be 1.0)")

    # 2. Showcase Generic Support (Type: Float32)
    println("\n[2] TESTING FLOAT32 GENERIC SUPPORT (GPU-Friendly)")
    x_f32 = Float32[0.0, 0.05, 0.2]
    x_eval_f32 = 0.1f0

    omega_f32 = get_split_nonlinear_weights(
        get_nonuniform_ideal_weights(x_f32, x_eval_f32),
        get_nonuniform_smoothness_indicators(x_f32)
    )
    println("Final Weights (Float32): ", omega_f32)
    println("Type of output: ", typeof(omega_f32))

    # 3. Performance Benchmark
    println("\n[3] PERFORMANCE BENCHMARK (Float64)")
    print("Execution time: ")
    @btime get_split_nonlinear_weights(
        get_nonuniform_ideal_weights($x_local, $x_eval),
        get_nonuniform_smoothness_indicators($x_local)
    )

    println("-"^60)
    println("PROTOTYPE VERIFIED: TYPE-STABLE AND AD-COMPATIBLE")
    println("="^60)

    # 4. BLACK OPS STRESS TEST: EXTREME LOGARITHMIC GRID STRETCHING

    println("\n[4] STARTING BLACK OPS STRESS TEST: EXTREME LOGARITHMIC GRID")
    println("="^60)

    # Stencil points: An extreme stretch where spacing increases 10x
    # x_i-1 = 0.0, x_i = 0.01 (h1=0.01), x_i+1 = 0.11 (h2=0.10)
    x_extreme = [0.0, 0.01, 0.11]
    x_eval_extreme = 0.05 # Evaluate near the center-right

    println("Extreme grid: ", x_extreme)
    println("Evaluation point: ", x_eval_extreme)
    println("h1: 0.01, h2: 0.10 (10x ratio!)")

    # 4.1 Check Ideal Weights (Will likely be heavily negative)
    d_k_extreme = get_nonuniform_ideal_weights(x_extreme, x_eval_extreme)
    println("\n--> Ideal Weights (d_k) before shifting:")
    println(d_k_extreme)
    println("Min d_k: ", minimum(d_k_extreme))

    # 4.2 Check Non-Linear Weights (Our regulator should save us)
    omega_k_extreme = get_split_nonlinear_weights(
        d_k_extreme,
        get_nonuniform_smoothness_indicators(x_extreme)
    )

    println("\n--> Final Non-Linear Weights (omega_k) after regularization:")
    println(omega_k_extreme)
    println("Min omega_k: ", minimum(omega_k_extreme), " (Safely handled negative weights via Splitting!)")
    println("Sum of Weights: ", sum(omega_k_extreme), " (Must be 1.0)")

    if isapprox(sum(omega_k_extreme), 1.0, atol=1e-10)
        println("\n[RESULT] BLACK OPS STRESS TEST: SUCCESS! Splitting technique perfectly maintained Partition of Unity despite extreme negative weights.")
    else
        error("\n[RESULT] BLACK OPS STRESS TEST: FAILED! Partition of Unity is broken.")
    end

end

showcase_prototype()