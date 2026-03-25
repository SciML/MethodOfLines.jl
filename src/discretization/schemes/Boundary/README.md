# Non-Uniform Boundary Stencils

This module provides strictly $O(1)$, zero-allocation mathematical engines for calculating one-sided finite difference weights on highly non-uniform grids.

## Mathematical Formulation

Consider a left boundary node $x_0$ and its two immediate interior neighbors $x_1$ and $x_2$. Let the grid spacings be:
* $h_1 = x_1 - x_0$
* $h_2 = x_2 - x_1$

We seek coefficients $c_0, c_1, c_2$ such that the derivative at the boundary is approximated by:
$$f^{(n)}(x_0) \approx c_0 f(x_0) + c_1 f(x_1) + c_2 f(x_2)$$

### First Derivative
For the first derivative ($n=1$), the Taylor series expansion on an irregular grid yields the following exact weights:
* $c_0 = -\frac{2h_1 + h_2}{h_1(h_1 + h_2)}$
* $c_1 = \frac{h_1 + h_2}{h_1 h_2}$
* $c_2 = -\frac{h_1}{h_2(h_1 + h_2)}$

### Second Derivative
For the second derivative ($n=2$), the weights are heavily asymmetric:
* $c_0 = \frac{2}{h_1(h_1 + h_2)}$
* $c_1 = -\frac{2}{h_1 h_2}$
* $c_2 = \frac{2}{h_2(h_1 + h_2)}$

## Implementation Details
* **Type Promotion:** Automatically handles mixed-type inputs (e.g., `Float32` and `Float64`) to prevent `MethodError` and ensure robust type stability.
* **AD Compatibility:** Fully supports `ForwardDiff.Dual` via generic `T<:Real` typing.
* **Symmetry:** Right boundary formulas mathematically reuse the left boundary logic with sign-flipping for odd derivatives to preserve DRY (Don't Repeat Yourself) principles.