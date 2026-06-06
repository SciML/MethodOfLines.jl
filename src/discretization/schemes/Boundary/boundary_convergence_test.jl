using Printf
include("boundary_weights.jl")

# MMS Test Functions: u(x) = exp(x)
# Analytical derivatives: u'(x) = exp(x), u''(x) = exp(x)
u_exact(x)  = exp(x)
du_exact(x)  = exp(x)
d2u_exact(x) = exp(x)

"""
    run_convergence_analysis(N_vals)

Performs a convergence study for the 4-point non-uniform boundary stencils 
using the Method of Manufactured Solutions (MMS).
"""
function run_convergence_analysis(N_vals)
    println("\n" * "="^85)
    println("   PR #539: Convergence Analysis (4-Point Boundary Stencils)")
    println("="^85)
    @printf("%-5s | %-18s | %-8s | %-18s | %-8s\n", "N", "1st Deriv Err", "Order", "2nd Deriv Err", "Order")
    println("-"^85)

    errs1 = Float64[]
    errs2 = Float64[]

    for N in N_vals
        # Grid setup: Stretched grid using power law
        ξ = range(0.0, 1.0, length=N)
        x = collect(@. ξ^1.3) 
        u = u_exact.(x)
        h1, h2, h3 = x[2]-x[1], x[3]-x[2], x[4]-x[3]

        # 1st Derivative Test
        c_1st = get_nonuniform_weights_1st_deriv_left_4pt(h1, h2, h3)
        du_num = sum(c_1st .* u[1:4])
        e1 = abs(du_num - du_exact(x[1]))
        push!(errs1, e1)

        # 2nd Derivative Test
        c_2nd = get_nonuniform_weights_2nd_deriv_left_4pt(h1, h2, h3)
        d2u_num = sum(c_2nd .* u[1:4])
        e2 = abs(d2u_num - d2u_exact(x[1]))
        push!(errs2, e2)

        if length(errs1) == 1
            @printf("%-5d | %.8e      | -        | %.8e      | -\n", N, e1, e2)
        else
            order1 = log2(errs1[end-1] / errs1[end])
            order2 = log2(errs2[end-1] / errs2[end])
            @printf("%-5d | %.8e      | %.2f     | %.8e      | %.2f\n", N, e1, order1, e2, order2)
        end
    end
    println("="^85)
end

run_convergence_analysis([32, 64, 128, 256, 512])