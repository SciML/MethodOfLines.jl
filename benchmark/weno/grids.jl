# Grid generators and quadrature-weighted error norms for the WENO benchmarks.
# Grid classes: :uniform (scalar dx, uniform kernel path); :uniform_vector (equispaced
# vector, non-uniform path, isolates kernel overhead); :stretched (smooth sinusoidal map);
# :perturbed (seeded random jitter).

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

"""
    grid_spec(kind, a, b, n)

Spatial discretization argument for `MOLFiniteDifference`: scalar step for `:uniform`,
node vector otherwise.
"""
function grid_spec(kind::Symbol, a, b, n)
    kind === :uniform && return (b - a) / (n - 1)
    kind === :uniform_vector && return uniform_grid(a, b, n)
    kind === :stretched && return stretched_grid(a, b, n)
    kind === :perturbed && return perturbed_grid(a, b, n)
    return error("unknown grid kind: $kind")
end

"""
    grid_nodes(kind, a, b, n)

Node vector realized by `grid_spec`.
"""
grid_nodes(kind::Symbol, a, b, n) =
    kind === :uniform ? uniform_grid(a, b, n) : grid_spec(kind, a, b, n)

min_spacing(kind::Symbol, a, b, n) = minimum(diff(grid_nodes(kind, a, b, n)))

"""
    front_adapted_grid(a, b, n; band, ratio = 10.0, w = 0.03)

Strictly increasing grid equidistributed w.r.t. the plateau density
`ρ(x) = 1 + (ratio - 1)·½·(tanh((x - band[1])/w) - tanh((x - band[2])/w))`,
concentrating nodes inside `band`. Inverse-CDF sampling on a 4001-point reference mesh.
"""
function front_adapted_grid(a, b, n; band, ratio = 10.0, w = 0.03)
    lo, hi = band
    @assert a < lo < hi < b
    ρ(x) = 1 + (ratio - 1) * (tanh((x - lo) / w) - tanh((x - hi) / w)) / 2

    xs = range(a, b; length = 4001)
    cdf = cumsum(ρ.(xs))
    cdf .-= cdf[1]
    cdf ./= cdf[end]

    # Inverse CDF by linear interpolation; monotone since ρ > 0.
    levels = range(0.0, 1.0; length = n)
    g = similar(collect(levels))
    j = 1
    for (i, ℓ) in enumerate(levels)
        while cdf[j + 1] < ℓ && j < length(xs) - 1
            j += 1
        end
        θ = (ℓ - cdf[j]) / (cdf[j + 1] - cdf[j])
        g[i] = xs[j] + θ * (xs[j + 1] - xs[j])
    end
    g[1] = a
    g[end] = b
    @assert all(diff(g) .> 0)
    return g
end

# Plain RMS is not a consistent L2 norm on non-uniform grids; composite trapezoid
# weights give ||e||² ≈ Σ wᵢeᵢ².

function trapezoid_weights(x::AbstractVector)
    n = length(x)
    w = zeros(float(eltype(x)), n)
    for i in 1:(n - 1)
        h = x[i + 1] - x[i]
        w[i] += h / 2
        w[i + 1] += h / 2
    end
    return w
end

l2_norm(e::AbstractVector, x::AbstractVector) = sqrt(sum(trapezoid_weights(x) .* abs2.(e)))

l2_error(u::AbstractVector, uexact::AbstractVector, x::AbstractVector) =
    l2_norm(u .- uexact, x)
