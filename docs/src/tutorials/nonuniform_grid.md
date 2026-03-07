# Finite Differences on Non-Uniform Grids

Many finite difference examples assume a **uniform spatial grid**, where the spacing between grid points is constant. However, in practical PDE problems, it is often useful to use **non-uniform grids**, where grid points are clustered in regions requiring higher resolution.

---

## Uniform Grid

A uniform grid has constant spacing:
$x_i = i \Delta x$

For the second derivative, we use the classical centered finite difference stencil:
$$u''(x_i) \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$

Example in Julia:
```julia
function second_derivative_uniform(u, dx, i)
    return (u[i+1] - 2u[i] + u[i-1]) / dx^2
end
Non-Uniform GridIn a non-uniform grid, the spacing varies. Let:$\Delta x_i = x_i - x_{i-1}$$\Delta x_{i+1} = x_{i+1} - x_i$The second derivative approximation becomes:$$u''(x_i) \approx \frac{2}{\Delta x_i (\Delta x_i + \Delta x_{i+1})} u_{i-1} - \frac{2}{\Delta x_i \Delta x_{i+1}} u_i + \frac{2}{\Delta x_{i+1} (\Delta x_i + \Delta x_{i+1})} u_{i+1}$$Example implementation:Juliafunction second_derivative_nonuniform(u, x, i)
    dx_i = x[i] - x[i-1]
    dx_ip1 = x[i+1] - x[i]

    return (
        2 / (dx_i * (dx_i + dx_ip1)) * u[i-1] -
        2 / (dx_i * dx_ip1) * u[i] +
        2 / (dx_ip1 * (dx_i + dx_ip1)) * u[i+1]
    )
end
Example GridAn example of a non-uniform grid array:Juliax = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
This grid places more points near the left boundary. Such grids are particularly useful when dealing with:Boundary layersSingularitiesAdaptive resolution requirementsSummaryUniform grids: Simpler stencils, easier implementation.Non-uniform grids: Flexible resolution, crucial for complex PDE problems, require modified finite difference operators.