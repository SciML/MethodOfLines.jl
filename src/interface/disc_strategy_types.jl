# Discretization strategies
# -------------------------
abstract type AbstractDiscretizationStartegy end
# The following are the discretization strategies that are currently implemented.
#
# Scalar discretization
# ~~~~~~~~~~~~~~~~~~~~~
# This is the default discretization strategy. It discretizes the PDESystem into a
# scalar ODEProblem. The discretization is done by discretizing the PDESystem
# into a set of ODEs for each index, and then concatenating them together. The resulting
# ODEProblem is then solved.
#
# This is the default strategy, and is used when the `discretization_strategy`
# keyword argument is not specified.
struct ScalarizedDiscretization <: AbstractDiscretizationStartegy end

# Array discretization
# ~~~~~~~~~~~~~~~~~~~~~
# This discretization strategy discretizes the PDESystem into an ArrayMaker of ODEs.
# This method means that Symbolics.build_function can generate looped code, which compiles
# much faster than the scalar method.
struct ArrayDiscretization <: AbstractDiscretizationStartegy end
