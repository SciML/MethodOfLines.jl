# [Pseudospectral Discretization](@id pseudospectral)

```julia
struct PseudospectralDiscretization <: AbstractEquationSystemDiscretization
    dxs::Any
    time::Any
    kwargs::Any
end
```

`PseudospectralDiscretization` discretizes a `PDESystem` by collocation on
spectral grids: every spatial derivative `Differential(x)^d` is replaced by the
dense differentiation matrix of the spectral interpolant in that direction, and
boundary conditions replace the equation at the boundary points, exactly as in
[`MOLFiniteDifference`](@ref molfd). This mirrors the handwritten
pseudospectral implementations in the
SciMLBenchmarks ([Allen–Cahn](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/SimpleHandwrittenPDE/allen_cahn_spectral_wpd/),
[Burgers](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/SimpleHandwrittenPDE/burgers_spectral_wpd/),
[KdV](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/SimpleHandwrittenPDE/kdv_spectral_wpd/) and
[Kuramoto–Sivashinsky](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/SimpleHandwrittenPDE/ks_spectral_wpd/)),
which are based on
`ClassicalOrthogonalPolynomials` Chebyshev collocation and
`SummationByPartsOperators` Fourier differentiation.

```julia
eq = [your system of equations, see examples for possibilities]
bcs = [your boundary conditions, see examples for possibilities]

domains = [your domain, a vector of Intervals i.e. x ∈ Interval(x_min, x_max)]

@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

discretization = PseudospectralDiscretization(dxs, t)
prob = discretize(pdesys, discretization)
```

Here `dxs` is a vector of pairs mapping each independent variable to a
collocation specification:

- `x => n::Integer` or `x => ChebyshevCollocation(n)`: discretize `x` on `n`
  Chebyshev–Lobatto points `x_k = a + (b - a)/2 * (1 - cos(π(k - 1)/(n - 1)))`.
  Use for non-periodic directions; truncating boundary conditions (Dirichlet,
  Neumann, Robin, higher-order) are applied as usual.
- `x => FourierCollocation(n)`: discretize `x` on `n` distinct equispaced
  points of the periodic interval `[a, b)`. A periodic boundary condition
  `u(t, a) ~ u(t, b)` is required for every dependent variable in that
  direction; derivative matching conditions such as
  `Differential(x)(u(t, a)) ~ Differential(x)(u(t, b))` are redundant - they
  hold identically for the trigonometric interpolant - and are skipped.
- `x => grid::AbstractVector`: a custom set of `n` collocation nodes;
  derivatives are the polynomial-interpolant (Fornberg) differentiation
  matrices on those nodes.

The second argument `time` is the continuous variable; if `time = nothing`,
discretization yields a `NonlinearProblem`.

## Example

```julia
using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq

@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# Heat equation on a Chebyshev-Lobatto grid
eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cospi(x / 2), u(t, -1) ~ 0.0, u(t, 1) ~ 0.0]
domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-1.0, 1.0)]
@named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

prob = discretize(pdesys, PseudospectralDiscretization([x => 32], t))
sol = solve(prob)
sol[u(t, x)] # solution on the collocation grid
```

## Notes

- Nested derivative terms such as `Differential(x)(a(u) * Differential(x)(u))`
  or `Differential(x)(u^2)` are evaluated directly by collocation, so no
  PDE-system transformation is performed for this discretization.
- Derivative BCs such as `Differential(x)(u(t, a)) ~ 0` are enforced through
  the differentiation matrix row at the boundary.
- Fourier directions are differentiated with the explicit trigonometric
  interpolant matrix raised to the requested order; all other grids use the
  maximal-stencil Fornberg weights, i.e. the polynomial interpolant.
- Spectral differentiation matrices are dense, so symbolic equation generation
  scales `O(n^2)` in the number of collocation points per direction, and the
  generated RHS is dense. This discretization is best for small `n` where
  spectral accuracy dominates, or for prototyping; for large systems or when
  sparsity matters, `MOLFiniteDifference` is more appropriate.
- `Integral` terms and multi-domain (interface) boundary conditions are not
  supported.
