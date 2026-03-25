# boundary_weights_prototype.jl

"""
Calculate 3-point one-sided forward finite difference weights
for a non-uniform grid at the left boundary (x_0).
h1 = x_1 - x_0  (Distance to first neighbor)
h2 = x_2 - x_1  (Distance to second neighbor)
"""
function get_nonuniform_boundary_weights_1st_deriv(h1::T, h2::T) where T<:AbstractFloat
    # Denominators (Paydalar)
    den1 = h1 * (h1 + h2)
    den2 = h1 * h2
    den3 = h2 * (h1 + h2)

    # Coefficients (c0 for x_0, c1 for x_1, c2 for x_2)
    c0 = -(2*h1 + h2) / den1
    c1 = (h1 + h2) / den2
    c2 = -h1 / den3

    return (c0, c1, c2)
end

# ==========================================
# 🚀 STRESS TEST & SHOWCASE
# ==========================================
function run_boundary_test()
    println("="^60)
    println("SCIML NON-UNIFORM BOUNDARY STENCIL PROTOTYPE")
    println("="^60)

    # 1. UNIFORM CASE (Eşit Aralıklı - Referans Testi)
    # Standart formül katsayıları: [-3/(2h), 4/(2h), -1/(2h)] olmalıdır.
    h_uniform = 0.1
    println("\n[1] TESTING UNIFORM GRID (h1 = 0.1, h2 = 0.1)")
    w_uniform = get_nonuniform_boundary_weights_1st_deriv(h_uniform, h_uniform)
    println("Calculated Weights (c0, c1, c2): ", w_uniform)
    
    expected_uniform = (-3/(2*h_uniform), 4/(2*h_uniform), -1/(2*h_uniform))
    println("Expected Weights (Classic):    ", expected_uniform)
    if all(w_uniform .≈ expected_uniform)
        println("✅ Uniform Check: PASSED!")
    end

    # 2. NON-UNIFORM CASE (Düzensiz Grid - Asıl Sınavımız)
    # Sınırda noktalar çok sıkışık (h1=0.01), sonra açılıyor (h2=0.1)
    println("\n[2] TESTING NON-UNIFORM GRID (h1 = 0.01, h2 = 0.1)")
    w_nonuniform = get_nonuniform_boundary_weights_1st_deriv(0.01, 0.1)
    println("Calculated Weights (c0, c1, c2): ", w_nonuniform)
    
    # Kural: Türev katsayılarının toplamı her zaman SIFIR olmalıdır! (Kütle korunumu)
    sum_weights = sum(w_nonuniform)
    println("Sum of non-uniform weights: ", sum_weights, " (Must be 0.0)")
    
    if abs(sum_weights) < 1e-10
        println("✅ Zero-Sum Mass Conservation: PASSED!")
    else
        println("❌ FAILED!")
    end

    # 3. SECOND DERIVATIVE TEST (Düzensiz Grid 2. Türev)
    println("\n[3] TESTING 2nd DERIVATIVE NON-UNIFORM (h1 = 0.01, h2 = 0.1)")
    w_2nd = get_nonuniform_boundary_weights_2nd_deriv(0.01, 0.1)
    println("Calculated 2nd Deriv Weights: ", w_2nd)
    
    sum_2nd = sum(w_2nd)
    println("Sum of 2nd deriv weights: ", sum_2nd, " (Must be 0.0 for constants)")
    
    if abs(sum_2nd) < 1e-10
        println("✅ 2nd Deriv Mass Conservation: PASSED!")
    else
        println("❌ FAILED!")
    end

    println("="^60)
end

"""
Calculate 3-point one-sided forward finite difference weights
for the SECOND derivative at the left boundary (x_0).
"""
function get_nonuniform_boundary_weights_2nd_deriv(h1::T, h2::T) where T<:AbstractFloat
    # Denominators
    den1 = h1 * (h1 + h2)
    den2 = h1 * h2
    den3 = h2 * (h1 + h2)

    # 2nd Derivative Coefficients
    c0 = 2.0 / den1
    c1 = -2.0 / den2
    c2 = 2.0 / den3

    return (c0, c1, c2)
end

run_boundary_test()