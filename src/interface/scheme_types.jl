abstract type AbstractScheme{AOrder} end

struct UpwindScheme <: AbstractScheme{1}
end

# Note: This type and its subtypes will become important later with the stencil interfaces as we will need to dispatch on derivative order and approximation order
