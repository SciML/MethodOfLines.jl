"""
`MOLMetadata`

A type used to store data about a PDESystem, and how it was discretized by MethodOfLines.jl.
Used to unpack the solution.

- `discretespace`: a DiscreteSpace object, used in the discretization.
- `disc`: a Discretization object, used in the discretization. Usually a
          MOLFiniteDifference object.
- `pdesys`: a PDESystem object, used in the discretization.
"""
struct MOLMetadata{hasTime, Ds,Disc,PDE} <: SciMLBase.AbstractDiscretizationMetadata{hasTime}
    discretespace::Ds
    disc::Disc
    pdesys::PDE
    function MOLMetadata(discretespace, disc, pdesys)
        if discretespace.time isa Nothing
            hasTime = Val(false)
        else
            hasTime = Val(true)
        end
        return new{hasTime, typeof(discretespace), typeof(disc), typeof(pdesys)}(discretespace,
                                                                                 disc, pdesys)
    end
end

#! Which package for methods, dispatch problem

#! where to intercept and wrap

#! prob.f.sys

#! 2d ODE nudge
