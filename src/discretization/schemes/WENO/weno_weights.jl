module WENOWeights

export get_nonuniform_ideal_weights, get_nonuniform_smoothness_indicators, get_nonuniform_nonlinear_weights

# 1. AUXILIARY MATHEMATICAL FUNCTIONS (PRIVATE)

"""
    lagrange_basis(x_local, k, x_eval)

Calculates the exact value of the k-th Lagrange basis polynomial.
Optimized for generic types (T) to support Automatic Differentiation (AD).
"""
function lagrange_basis(x_local::AbstractVector{T}, k::Int, x_eval::T) where T<:Real
    n = length(x_local)
    L = one(T)  # Ensures L is the same type as input (Float32, Float64, Dual, etc.)
    for j in 1:n
        if j != k
            L *= (x_eval - x_local[j]) / (x_local[k] - x_local[j])
        end
    end
    return L
end

# 2. MAIN API FUNCTIONS (PUBLIC)

"""
    get_nonuniform_ideal_weights(x_local, x_eval)

Dynamically calculates the ideal weights (d_k). 
Type-stable and AD-compatible.
"""
function get_nonuniform_ideal_weights(x_local::AbstractVector{T}, x_eval::T) where T<:Real
    n = length(x_local)
    weights = zeros(T, n) # Pre-allocate with type T
    
    for k in 1:n
        weights[k] = lagrange_basis(x_local, k, x_eval)
    end
    
    return weights
end

"""
    get_nonuniform_smoothness_indicators(x_local)

Calculates modified smoothness indicators (β_k). 
Uses eltype(x_local) to maintain precision across different float types.
"""
function get_nonuniform_smoothness_indicators(x_local::AbstractVector{T}) where T<:Real
    n = length(x_local)
    
    if n == 3
        h1 = x_local[2] - x_local[1]
        h2 = x_local[3] - x_local[2]
        
        # Geometrically derived factor for non-uniform stencils
        w_dynamic = (h1^2 + h1*h2 + h2^2) / (h1 + h2)^2
        return fill(w_dynamic, n)
    else
        return ones(T, n)
    end
end

"""
    get_nonuniform_nonlinear_weights(d_k, beta_k; epsilon=1e-6)

Calculates final non-linear WENO weights (ω_k) with Negative Weight Treatment.
Fully generic and optimized for zero-cost abstraction.
"""
function get_nonuniform_nonlinear_weights(d_k::AbstractVector{T}, beta_k::AbstractVector{T}; 
                                         epsilon::Real=1e-6) where T<:Real
    n = length(d_k)
    eps_T = T(epsilon) # Convert epsilon to type T to prevent promotion overhead
    
    # --- NEGATIVE WEIGHT REGULARIZATION ---
    min_d = minimum(d_k)
    # Using a local safe copy that respects type T
    d_k_safe = copy(d_k)
    
    if min_d < zero(T)
        shift = abs(min_d) + T(1e-4)
        d_k_safe .= d_k_safe .+ shift
        d_k_safe ./= sum(d_k_safe)
    end
    
    # Step 1: Calculate unscaled alpha_k
    # We use similar(d_k) to ensure the output vector matches input type
    alpha_k = similar(d_k)
    for k in 1:n
        alpha_k[k] = d_k_safe[k] / (eps_T + beta_k[k])^2
    end
    
    # Step 2: Normalize to get omega_k
    sum_alpha = sum(alpha_k)
    omega_k = alpha_k ./ sum_alpha
    
    return omega_k
end

end # module