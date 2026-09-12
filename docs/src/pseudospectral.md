# [Pseudospectral Discretization](@id pseudospectral)

`PseudospectralDiscretization` discretizes a `PDESystem` by collocation on spectral
grids. The unknowns are the values of each dependent variable at the collocation
nodes, exactly as for [`MOLFiniteDifference`](@ref molfd); the difference is the
derivative approximation. Every spatial derivative `Differential(x)^d` is the
derivative of the spectral interpolant through the nodes in that direction, so a
derivative at one node involves the values at all nodes of the direction rather
than a fixed stencil. Nonlinear terms are evaluated pointwise on the grid, and the
unknowns stay in physical (grid) space. With an `AbstractFFTs` backend such as
`FFTW` loaded, derivatives are applied with FFTs in both Fourier and Chebyshev
directions; otherwise the equivalent dense differentiation matrix is used. See
[FFT derivatives](@ref pseudospectral-fft).

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
  Chebyshev–Lobatto points `x_k = a + (b - a)/2 * (1 - cos(π(k - 1)/(n - 1)))`,
  which include both endpoints. Derivatives are those of the degree `n - 1`
  polynomial interpolant. Use for non-periodic directions.
  `ChebyshevCollocation(n; fft = false)` keeps the dense matrix even when an FFT
  backend is loaded.
- `x => FourierCollocation(n)`: discretize `x` on `n` distinct equispaced points
  of the periodic interval `[a, b)`. Derivatives are those of the trigonometric
  interpolant. A periodic boundary condition `u(t, a) ~ u(t, b)` is required for
  every dependent variable in that direction. `FourierCollocation(n; fft = false)`
  keeps the dense matrix even when an FFT backend is loaded.
- `x => grid::AbstractVector`: a custom set of `n` nodes spanning the domain;
  derivatives are those of the polynomial interpolant through the nodes (Fornberg
  weights over the whole grid). Clustered nodes such as Chebyshev points are needed
  for this to be well conditioned; equispaced nodes suffer the Runge phenomenon.

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

## Boundary conditions on Chebyshev directions

A Chebyshev–Lobatto grid contains both endpoints, and the boundary values are
unknowns of the discretized system. The PDE is imposed at the interior nodes. At
each boundary node, the PDE is not imposed; the equation for that node is the
boundary condition itself, written in terms of the grid values. Concretely, for
`u(t, x)` on `n` nodes with one condition at each end, the discretized system is
`n - 2` interior equations plus the two boundary equations, with `n` unknowns.

Derivatives in a boundary condition are the boundary rows of the differentiation
matrices: `Differential(x)(u(t, a))` becomes the first row of the first-derivative
matrix dotted with all `n` values in that direction, which is the exact derivative
of the interpolant at the endpoint. This makes the following conditions work
without any special handling, on their own or combined at the same boundary:

- Dirichlet, constant or time-dependent: `u(t, a) ~ 1.0`, `u(t, a) ~ sin(t)`.
- Neumann: `Differential(x)(u(t, a)) ~ 0.0`.
- Robin and nonlinear conditions: `Differential(x)(u(t, a)) + u(t, a) ~ 0.0`,
  `Differential(x)(u(t, b)) ~ u(t, b)^2`.
- Higher-order conditions: `(Differential(x)^2)(u(t, a)) ~ 0.0`, for any derivative
  order below the number of nodes.
- Conditions on other variables of a coupled system, and, in more than one
  dimension, conditions varying along the boundary face such as
  `u(t, a, y) ~ sin(y)`.

Each boundary condition at an end of a direction removes the PDE from one more
node at that end. A second-order equation takes one condition per end. A
fourth-order equation such as a clamped beam takes two, e.g. `u(t, a) ~ 0.0` and
`Differential(x)(u(t, a)) ~ 0.0`, and the PDE is then imposed on `n - 4` nodes. A
first-order equation such as advection takes a condition at the inflow end only:
where no condition is given, the PDE is imposed at the boundary node itself.

Because the boundary values are unknowns constrained algebraically, the
time-dependent system is a differential-algebraic system of index 1.
[`discretize`](@ref) returns a `DAEProblem` with `BrownFullBasicInit()`, which
solves the boundary equations for the boundary values at the initial time; the
initial condition is used at the interior nodes only, so a mismatch between the
initial condition and the boundary conditions at the corners of the domain is
tolerated. The compiled `ODEProblem` path (`symbolic_discretize` followed by
`mtkcompile`) instead eliminates the boundary unknowns: Dirichlet values are
substituted, and derivative conditions are solved for the boundary value in terms
of the interior values.

Boundary conditions must sit at the ends of the domain. Periodic conditions are not
accepted on a Chebyshev direction: use `FourierCollocation` for that direction.
Multi-domain interface conditions between different variables are not supported.

## Boundary conditions on Fourier directions

`FourierCollocation(n)` stores `n + 1` grid points, the last being the same
physical node as the first, so that the solution interface reports the value at
both ends of the domain. The upper endpoint is an algebraic unknown pinned by the
periodic condition `u(t, a) ~ u(t, b)`, which is the only condition the direction
needs. Derivatives never involve the duplicated node: each differentiation row acts
on the `n` distinct values, and the row for the upper endpoint is the row of the
lower one.

Derivative matching conditions such as
`Differential(x)(u(t, a)) ~ Differential(x)(u(t, b))` hold identically for the
trigonometric interpolant. They are checked and skipped, so they may be included
for readability but add no equations. A matching condition with a jump, such as
`Differential(x)(u(t, a)) ~ Differential(x)(u(t, b)) + 1`, cannot be represented
and raises an error. Truncating conditions such as Dirichlet values are rejected on
a Fourier direction.

In more than one dimension, the corner grid points at the duplicated index of a periodic
direction are not connected to the interior and are pinned to zero by the corner
equations, as on the finite difference path. They are visible only as those corner
entries of `sol[u(t, x, y)]`.

## Array form and scaling

The interior of each PDE is emitted as a single symbolic array equation over slices
of the discretized variables, in the same way as the
[array form of `MOLFiniteDifference`](@ref molfd). Each spatial derivative is an
opaque operator holding the rows of the differentiation matrix for the interior
nodes, applied along its axis to the full slice of its argument:

```julia
Differential(t)(u[2:n-1]) ~ SpectralApply(D2[2:n-1, :])(u[1:n]) - u[2:n-1] .^ 3
```

Nonlinear terms broadcast over the slices, and nested derivative terms evaluate
their argument on the whole direction before applying the outer operator, so
`Differential(x)(a(u) * Differential(x)(u))` becomes
`D1 * (a.(u) .* (D1 * u))` on slices. Boundary conditions on the faces of a
multi-dimensional domain are array equations as well, so the number of symbolic
equations is independent of the resolution in any number of spatial dimensions.
The differentiation operators are held inside the `SpectralApply` terms rather than
written into the expression tree, so the size of the symbolic expressions is
independent of the resolution too. Stationary systems and equations containing
patterns without a slice form fall back to one scalar equation per grid point,
each carrying a full differentiation row.

The `DAEProblem` returned by [`discretize`](@ref) keeps this array form at the
`System` level. ModelingToolkit's code generation then scalarizes the residual into
one entry per unknown, each referencing the shared derivative terms, which the
generated code evaluates once per call; the compile time therefore still grows
with the resolution, as it does for `MOLFiniteDifference` (see
[MethodOfLines.jl#691](https://github.com/SciML/MethodOfLines.jl/issues/691)).

Without an FFT backend each derivative costs a dense matrix product: `O(n^2)` per
application in one dimension and `O(n^2 m)` along the first axis of an `n × m`
grid. The Jacobian is dense either way.

## [FFT derivatives](@id pseudospectral-fft)

Loading an `AbstractFFTs` backend, in practice

```julia
using FFTW
```

activates the `MethodOfLinesAbstractFFTsExt` extension. From then on the
whole-direction derivatives of the array form are applied with FFTs along their
axis, at `O(n log n)` instead of `O(n^2)` per application:

- `FourierCollocation`: `irfft((ik)^d .* rfft(u))`, with the Nyquist mode zeroed so
  that the result equals the dense matrix `D1^d` to roundoff.
- `ChebyshevCollocation`: the Chebyshev transform. The DCT-I of each line is the
  real FFT of its even extension, the derivative's Chebyshev coefficients follow
  from the backward recurrence `b[k-1] = b[k+1] + 2k a[k]` applied `d` times, and
  the same transform maps the coefficients back to the nodes. This equals the
  polynomial-interpolant differentiation matrix to roundoff. The derivative is
  computed on every node, and the interior equation keeps the interior rows by
  slicing; the boundary rows are what the boundary conditions replace.

The Chebyshev transform of `n` nodes is a real FFT of length `2(n - 1)`, so
`n = 2^k + 1` gives the fastest transforms. FFT plans are created on first use and
cached per array size. Jacobian evaluations
through dual numbers, boundary rows in boundary conditions and pinned single rows
use the dense matrix, which is kept alongside the FFT operator.
`ChebyshevCollocation(n; fft = false)` and `FourierCollocation(n; fft = false)` opt
a direction out; custom node vectors always use the dense matrix.

## Limitations

- `Integral` terms are not supported.
- Multi-domain (interface) boundary conditions are not supported; only
  same-variable periodic conditions with `FourierCollocation`.
- There is no upwinding or scheme selection: all derivative orders in a direction
  share the same spectral differentiation matrix. Discontinuous solutions produce
  Gibbs oscillations, as with any global spectral method.
- The Jacobian is dense, so implicit time stepping costs `O(n^3)` per factorization
  in one dimension regardless of how the derivatives are applied.
