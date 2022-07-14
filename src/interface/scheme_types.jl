abstract type AbstractScheme{DOrder, AOrder} end

struct UpwindScheme <: AbstractScheme{1, 1}
end

struct WENOScheme <: AbstractScheme{1, 5}
end
