@testset "User defined scheme, 3 interior points" begin
    f_interior = (u, p, t, x, dx) -> IfElse.ifelse(
        u[2] < 0, (u[2] - u[1]) / dx, (u[3] - u[2]) / dx)
    f_lower = (u, p, t, x, dx) -> (u[2] - u[1]) / dx
    f_upper = (u, p, t, x, dx) -> (u[2] - u[1]) / dx

    scheme = FunctionScheme{3, 2}(f_interior, f_lower, f_upper, false)

    disc = MOLFiniteDifference([x => dx], t, advection_scheme = scheme)
end
