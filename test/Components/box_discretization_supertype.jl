using MethodOfLines, ModelingToolkit, DomainSets, Test
using MethodOfLines: PDEBase

@parameters t x
dx = 0.1

@testset "Finite difference and pseudospectral discretizations are box discretizations" begin
    @test MOLFiniteDifference([x => dx], t) isa PDEBase.AbstractBoxDiscretization
    @test PseudospectralDiscretization([x => 8], t) isa PDEBase.AbstractBoxDiscretization
    @test !(MOLFiniteDifference([x => dx], t) isa PDEBase.AbstractMeshDiscretization)
end

# M0 exit (FEM spec section 9): a Rectangle problem with trace conditions, lowered by
# `PDEBase.to_box_bcs` and solved by MOLFiniteDifference. Pending until PDEBase ships `to_box_bcs`.
@testset "to_box_bcs lowering of Rectangle trace conditions" begin
    if isdefined(PDEBase, :to_box_bcs)
        @test_broken false # write the Rectangle problem here once the lowering exists
    else
        @test_skip isdefined(PDEBase, :to_box_bcs)
    end
end
