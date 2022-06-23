using Test, LinearAlgebra
using MethodOfLines

@testset "finite-difference weights from fornberg(1988) & fornberg(2020)" begin
    order = 2
    z = 0.0
    x = [-1, 0, 1.0]
    @test MethodOfLines.calculate_weights(order, z, x) == [1, -2, 1]  # central difference of second-derivative with unit-step

    order = 1
    z = 0.0
    x = [-1.0, 1.0]
    @test MethodOfLines.calculate_weights(order, z, x) == [-0.5, 0.5] # central difference of first-derivative with unit step

    order = 1
    z = 0.0
    x = [0, 1]
    @test MethodOfLines.calculate_weights(order, z, x) == [-1, 1] # forward difference

    order = 1
    z = 1.0
    x = [0, 1]
    @test MethodOfLines.calculate_weights(order, z, x) == [-1, 1] # backward difference

    # forward-diff of third derivative with order of accuracy == 3
    order = 3
    z = 0.0
    x = [0, 1, 2, 3, 4, 5]
    @test MethodOfLines.calculate_weights(order, z, x) ==
          [-17 / 4, 71 / 4, -59 / 2, 49 / 2, -41 / 4, 7 / 4]

    order = 3
    z = 0.0
    x = collect(-3:3)
    d, e = MethodOfLines.calculate_weights(order, z, x; dfdx = true)
    @test d ≈ [-167 / 18000, -963 / 2000, -171 / 16, 0, 171 / 16, 963 / 2000, 167 / 18000]
    @test e ≈ [-1 / 600, -27 / 200, -27 / 8, -49 / 3, -27 / 8, -27 / 200, -1 / 600]

    order = 3
    z = 0.0
    x = collect(-4:4)
    d, e = MethodOfLines.calculate_weights(order, z, x; dfdx = true)
    @test d ≈ [
        -2493 / 5488000,
        -12944 / 385875,
        -87 / 125,
        -1392 / 125,
        0,
        1392 / 125,
        87 / 125,
        12944 / 385875,
        2493 / 5488000,
    ]
    @test e ≈ [
        -3 / 39200,
        -32 / 3675,
        -6 / 25,
        -96 / 25,
        -205 / 12,
        -96 / 25,
        -6 / 25,
        -32 / 3675,
        -3 / 39200,
    ]
end
