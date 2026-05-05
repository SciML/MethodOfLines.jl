# Non-Uniform Boundary Stencils

This module provides strictly $O(1)$, zero-allocation mathematical engines 
for calculating high-order, one-sided finite difference weights on highly 
non-uniform grids.

## Mathematical Formulation

The module implements 4-point stencils to achieve higher-order accuracy 
compared to standard boundary treatments. Consider a left boundary node 
$x_0$ and its three immediate interior neighbors $x_1, x_2, x_3$. 
Let the local grid spacings be:
* $h_1 = x_1 - x_0$
* $h_2 = x_2 - x_1$
* $h_3 = x_3 - x_2$

We seek coefficients $c_0, c_1, c_2, c_3$ such that the derivative at the 
boundary is approximated by the summation:
$$f^{(n)}(x_0) \approx \sum_{i=0}^{3} c_i f(x_i)$$

The weights are derived via Lagrange interpolating polynomials on 
irregular intervals, ensuring exactness for polynomials up to degree 3 
for the first derivative and degree 2 for the second derivative.

## Implementation Details

* **Numerical Stability:** Uses subtraction-free logic for denominators 
  by constructing them directly from intervals ($h_i$). This minimizes 
  floating-point reconstruction noise inherited from absolute coordinate 
  differences.
* **Mass Conservation:** Employs Kahan (compensated) summation logic 
  to calculate the boundary coefficient $c_0$. This ensures the property 
  $\sum c_i = 0$ is maintained at the machine epsilon limit, preserving 
  stability in long-term simulations.
* **Type Promotion:** Automatically handles mixed-type inputs (e.g., 
  `Float32` and `Float64`) to prevent `MethodError` and ensure robust 
  type stability.
* **AD Compatibility:** Fully supports forward-mode automatic 
  differentiation via generic `T<:Real` typing, making it compatible 
  with `ForwardDiff.jl`.
* **Symmetry:** Right boundary formulas mathematically reuse the left 
  boundary logic with appropriate sign-flipping to preserve 
  DRY (Don't Repeat Yourself) principles.