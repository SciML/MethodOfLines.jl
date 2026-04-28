module WENOWeights

export get_nonuniform_ideal_weights, get_nonuniform_smoothness_indicators, get_split_nonlinear_weights

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
    get_split_nonlinear_weights(d_k, beta_k; epsilon=1e-6)

Implements the Shi-Hu-Shu (2002) splitting technique with proper scaling factors.
Splits the weights into positive sets and recombines them using their mass sums (sigma)
to guarantee high-order accuracy without negative weight instabilities.
"""
function get_split_nonlinear_weights(d_k::AbstractVector{T}, beta_k::AbstractVector{T}; 
                                     epsilon::Real=1e-6) where T<:Real
    n = length(d_k)
    eps_T = T(epsilon)
    theta = T(3.0) # Splitting factor

    # 1. SPLIT THE WEIGHTS
    d_plus = similar(d_k)
    d_minus = similar(d_k)
    for k in 1:n
        d_plus[k] = zero(T) + T(0.5) * (d_k[k] + theta * abs(d_k[k]))
        d_minus[k] = d_plus[k] - d_k[k]
    end

    sigma_plus = sum(d_plus)
    sigma_minus = sum(d_minus)

    # 2. CALCULATE OMEGA PLUS
    alpha_plus = similar(d_k)
    for k in 1:n
        alpha_plus[k] = d_plus[k] / (eps_T + beta_k[k])^2
    end
    omega_plus = alpha_plus ./ sum(alpha_plus)

    # 3. CALCULATE OMEGA MINUS
    alpha_minus = similar(d_k)
    for k in 1:n
        alpha_minus[k] = d_minus[k] / (eps_T + beta_k[k])^2
    end
    omega_minus = alpha_minus ./ sum(alpha_minus)

    omega_k = similar(d_k)
    for k in 1:n
        omega_k[k] = sigma_plus * omega_plus[k] - sigma_minus * omega_minus[k]
    end

    return omega_k
end

end # module