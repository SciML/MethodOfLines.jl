abstract type AbstractScheme end

struct UpwindScheme <: AbstractScheme
    order
    function UpwindScheme(approx_order = 1)
        return new(approx_order)
    end
end

extent(scheme::UpwindScheme, dorder) = dorder+scheme.order-1

"""
`WENOScheme` of Jiang and Shu
## Keyword Arguments
- `epsilon`: A quantity used to prevent vanishing denominators in the scheme, defaults to `1e-6`. More sensetive problems will benefit from a smaller value.
"""
struct WENOScheme <: AbstractScheme
    epsilon
    function WENOScheme(epsilon = 1e-6)
        new(epsilon)
    end
end

function extent(::WENOScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for WENO schemes."
    return 2
end

# Note: This type and its subtypes will become important later with the stencil interfaces as we will need to dispatch on derivative order and approximation order

# Functional Schemes

"""
    # FunctionalScheme
A user definable scheme that takes a set of functions as input. The functions define the derivative at the interior, lower boundary, and upper boundary, taking the following inputs:

Functions must be of the form `f(u, p, t, deriv_iv, d_iv)`.

For the interior, `u` takes a vector of dependent variable values in the direction of the derivative
of length `interior_points`. `interior_points` must be odd, as this function defines the derivative at the center of the input points.

For the lower and upper boundaries, `u` takes a vector of dependent variable values of length `boundary_points`. This will be the `boundary_points` number of points closest to the lower and upper boundary respectively. The boundary functions define the derivative at their index in the function vector, numbering from the boundary. For example, if `boundary_points = 3`, the first function in the vector will define the derivative at the boundary, the second at the boundary plus one step, and the third at the boundary plus two steps.
In general, `upper` and `lower` must be at least `floor(interior_points/2)` long. Where you have no good approximation for a derivative at the boundary, you can use `nothing` as a placeholder. MethodOfLines will then attempt to use an extrapolation here where nessesary.

`p` will take all parameter values in the order specified in the PDESystem.

`deriv_iv` takes a vector of independent variable values of the same support as for `u`, for the independent variable in the direction of the derivative.

If `is_nonuniform` is false, `d_iv` will take a scalar value of the stepsize between the points used to call the function in `u` and `deriv_iv`.

If `is_nonuniform` is true, the scheme must be able to accept `d_iv` as a vector of stepsizes between the points used to call the function in `u` and `deriv_iv`, therefore of length `length(u)-1`. If it is used on a uniform grid, the stepsizes will be the same for all points.
"""
struct FunctionalScheme{F, V1, V2} <: AbstractScheme
    """
    The function that defines the scheme on the interior.
    """
    interior::F
    """
    The vector of functions that defines the scheme near the lower boundary.
    """
    lower::V1
    """
    The vector functions that defines the scheme near the upper boundary.
    """
    upper::V2
    """
    The number of interior points that the interior scheme takes as input
    """
    interior_points::Int
    """
    The number of boundary points that the lower and upper schemes take as input.
    """
    boundary_points::Int
    """
    Whether this scheme takes grid steps as input for the nonuniform case.
    """
    is_nonuniform::Bool
end
