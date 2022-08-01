abstract type AbstractScheme end

struct UpwindScheme <: AbstractScheme
    order
    function UpwindScheme(approx_order = 1)
        return new(approx_order)
    end
end

extent(::UpwindScheme, dorder) = dorder

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
