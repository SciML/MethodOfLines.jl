abstract type AbstractScheme end

struct UpwindScheme <: AbstractScheme
    order
    function UpwindScheme(approx_order=1)
        return new(approx_order)
    end
end

extent(scheme::UpwindScheme, dorder) = dorder + scheme.order - 1

# Functional Schemes

"""
    # FunctionalScheme
A user definable scheme that takes a set of functions as input. The functions define the derivative at the interior, lower boundary, and upper boundary.

`lower` and `upper` should be vectors of functions. In general, `upper` and `lower` must be at least `floor(interior_points/2)` long. Where you have no good approximation for a derivative at the boundary, you can use `nothing` as a placeholder. MethodOfLines will then attempt to use an extrapolation here where nessesary. Be warned that this can lead to instability.

The boundary functions define the derivative at their index in the function vector, numbering from the boundary. For example, if `boundary_points = 3`, the first function in the vector will define the derivative at the boundary, the second at the boundary plus one step, and the third at the boundary plus two steps.

The functions making up the taking the following inputs:

Functions must be of the form `f(u, p, t, deriv_iv, d_iv)`.

For the interior, `u` takes a vector of dependent variable values in the direction of the derivative
of length `interior_points`. `interior_points` must be odd, as this function defines the derivative at the center of the input points.

For the lower and upper boundaries, `u` takes a vector of dependent variable values of length `boundary_points`. This will be the `boundary_points` number of points closest to the lower and upper boundary respectively.
`p` will take all parameter values in the order specified in the PDESystem, with the scheme's parameters prepended to the list.

`deriv_iv` takes a vector of independent variable values of the same support as for `u`, for the independent variable in the direction of the derivative.

If `is_nonuniform` is false, `d_iv` will take a scalar value of the stepsize between the points used to call the function in `u` and `deriv_iv`.

If `is_nonuniform` is true, the scheme must be able to accept `d_iv` as a vector of stepsizes between the points used to call the function in `u` and `deriv_iv`, therefore of length `length(u)-1`. A method should also be defined for the case where `d_iv` is a scalar, in which case the stepsizes are assumed to be uniform.
"""
struct FunctionalScheme{F, V1, V2, V3} <: AbstractScheme
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
    """
    parameters for the scheme, should be a vector of numbers.
    """
    ps::V3
    """
    The name of the scheme.
    """
    name::String
end

function FunctionalScheme{ips, bps}(interior, lower, upper, is_nonuniform = false, ps = []; name = "FunctionalScheme") where {ips, bps}

    FunctionalScheme{typeof(interior), typeof(lower),
                     typeof(upper), typeof(ps)}(interior, lower, upper,
                                                ips, bps, is_nonuniform, ps)
end

function extent(scheme::FunctionalScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for functional schemes."
    lower = length(findall(isnothing, scheme.lower))
    upper = length(findall(isnothing, scheme.upper))
    @assert lower == upper "Scheme must have symmetric extent; same number of placeholders in lower and upper boundary functions."
    return lower
end

function lower_extent(scheme::FunctionalScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for functional schemes."
    return length(findall(isnothing, scheme.lower))
end

function upper_extent(scheme::FunctionalScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for functional schemes."
    return length(findall(isnothing, scheme.upper))
end
