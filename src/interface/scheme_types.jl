abstract type AbstractScheme{AOrder} end

struct UpwindScheme <: AbstractScheme{1}
end

extent(::UpwindScheme, dorder) = dorder

struct WENOScheme <: AbstractScheme{5}
end

function extent(::WENOScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for WENO schemes."
    return 2
end

approximation_order(::AbstractScheme{Aorder}) where Aorder = Aorder
# Note: This type and its subtypes will become important later with the stencil interfaces as we will need to dispatch on derivative order and approximation order
