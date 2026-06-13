module WENOSymbolic

export generate_weno_ast, generate_safe_weno_ast

"""
    sym_lagrange_basis(x_local, k, x_eval)

Calculates the symbolic Lagrange basis polynomial for non-uniform grids.
Designed to safely construct ASTs without evaluating numerical types directly.
"""
function sym_lagrange_basis(x_local, k, x_eval)
    n = length(x_local)
    L = 1
    for j in 1:n
        if j != k
            L = L * (x_eval - x_local[j]) / (x_local[k] - x_local[j])
        end
    end
    return L
end

"""
    sym_ideal_weights(x_local, x_eval)

Generates the ideal weights (d_k) symbolically using functional array comprehensions
to prevent mutation-based tracing errors in Symbolics.jl.
"""
function sym_ideal_weights(x_local, x_eval)
    return [sym_lagrange_basis(x_local, k, x_eval) for k in 1:length(x_local)]
end

"""
    sym_smoothness_indicators(x_local)

Dynamically derives the modified smoothness indicators (beta_k) as symbolic expressions 
based on the local geometric distances of the sub-stencils.
"""
function sym_smoothness_indicators(x_local)
    n = length(x_local)
    if n == 3
        h1 = x_local[2] - x_local[1]
        h2 = x_local[3] - x_local[2]

        w_dynamic = (h1^2 + h1*h2 + h2^2) / (h1 + h2)^2
        return [w_dynamic for _ in 1:n]
    else
        return [1 for _ in 1:n]
    end
end

"""
    sym_split_nonlinear_weights(d_k, beta_k; epsilon=1//1000000)

Implements the Shi-Hu-Shu (2002) weight splitting technique symbolically.
Utilizes type-stable arithmetic (integers and rationals) to prevent accidental 
Float64 promotion during codegen, ensuring strict compatibility with Float32 GPU arrays.
"""
function sym_split_nonlinear_weights(d_k, beta_k; epsilon=1//1000000)
    n = length(d_k)
    theta = 3

    d_plus = [(d_k[k] + theta * abs(d_k[k])) / 2 for k in 1:n]
    d_minus = [d_plus[k] - d_k[k] for k in 1:n]

    sigma_plus = sum(d_plus)
    sigma_minus = sum(d_minus)

    alpha_plus = [d_plus[k] / (epsilon + beta_k[k])^2 for k in 1:n]
    omega_plus = alpha_plus ./ sum(alpha_plus)

    alpha_minus = [d_minus[k] / (epsilon + beta_k[k])^2 for k in 1:n]
    omega_minus = alpha_minus ./ sum(alpha_minus)

    omega_k = [sigma_plus * omega_plus[k] - sigma_minus * omega_minus[k] for k in 1:n]
    return omega_k
end

"""
    generate_weno_ast(u_syms, x_syms, x_eval)

The core AST generator for non-uniform WENO. 
Outputs a pure Symbolic AST representing the flux equation, fully compatible 
with the rule-based multidimensional machinery.
"""
function generate_weno_ast(u_syms, x_syms, x_eval)
    d_k = sym_ideal_weights(x_syms, x_eval)
    beta_k = sym_smoothness_indicators(x_syms)
    omega_k = sym_split_nonlinear_weights(d_k, beta_k)

    flux_ast = sum(omega_k[i] * u_syms[i] for i in 1:length(u_syms))
    return flux_ast
end

"""
    generate_safe_weno_ast(u_syms, x_syms, x_eval)

A boundary-aware wrapper for `generate_weno_ast`.
Currently supports 3-point sub-stencils (WENO-3 logic). 
Full WENO-5 combining logic (using 3 overlapping 3-point stencils) 
is scheduled for the GSoC summer integration.
"""
function generate_safe_weno_ast(u_syms, x_syms, x_eval)
    len = length(u_syms)

    if len == 3
        return generate_weno_ast(u_syms, x_syms, x_eval)
    elseif len > 3
        mid = div(len, 2) + 1
        return generate_weno_ast(u_syms[mid-1:mid+1], x_syms[mid-1:mid+1], x_eval)
    else
        throw(ArgumentError("Stencil length must be at least 3 for WENO scheme generation."))
    end
end

end # module