# Grid generators for the WENO benchmarks. :uniform is a scalar dx (uniform kernel
# path); :uniform_vector, :stretched and :perturbed are node vectors through the
# non-uniform path.

using StableRNGs

const GRID_KINDS = (:uniform, :uniform_vector, :stretched, :perturbed)

uniform_grid(a, b, n) = collect(range(a, b; length = n))

# Same generator as the accuracy test suite.
stretched_grid(a, b, n; amp = 0.05) = [
    let ξ = a + (b - a) * (i - 1) / (n - 1)
            ξ + amp * sinpi(2 * (ξ - a) / (b - a))
    end
        for i in 1:n
]

function perturbed_grid(a, b, n; rel_amp = 0.3, seed = 1234)
    rng = StableRNG(seed)
    g = uniform_grid(a, b, n)
    h = (b - a) / (n - 1)
    g[2:(end - 1)] .+= (2 .* rand(rng, n - 2) .- 1) .* (rel_amp * h / 2)
    @assert all(diff(g) .> 0)
    return g
end

# Spatial discretization argument for `MOLFiniteDifference`: scalar step for :uniform,
# node vector otherwise.
function grid_spec(kind::Symbol, a, b, n)
    kind === :uniform && return (b - a) / (n - 1)
    kind === :uniform_vector && return uniform_grid(a, b, n)
    kind === :stretched && return stretched_grid(a, b, n)
    kind === :perturbed && return perturbed_grid(a, b, n)
    return error("unknown grid kind: $kind")
end

grid_nodes(kind::Symbol, a, b, n) =
    kind === :uniform ? uniform_grid(a, b, n) : grid_spec(kind, a, b, n)

min_spacing(kind::Symbol, a, b, n) = minimum(diff(grid_nodes(kind, a, b, n)))
