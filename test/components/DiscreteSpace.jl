using ModelingToolkit, MethodOfLines, DomainSets, Test

@parameters x, y, t 
@variables u(..) v(..)

indvars = [x, y, t]
nottime = [x, y]
depvars = [u, v]

t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.
y_min = 0.
y_max = 2.
dx = 0.1; dy = 0.2
order = 2

domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max), y ∈ Interval(y_min,y_max)]

@testset "Test 01: discretization of variables, center aligned grid" begin
    # Test centered order
    disc = MOLFiniteDifference([x=>dx, y=>dy], t; centered_order=order)

    s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, nottime, disc)

    discx = x_min:dx:x_max
    discy = y_min:dy:y_max

    @test s.grid[x] == discx 
    @test s.grid[y] == discy

    @test s.axies[x] == s.grid[x]
    @test s.axies[y] == s.grid[y]

    @test s.discvars[u] == collect(first(@variables u[axes(discx), axes(discy)]))
    @test s.discvars[v] == collect(first(@variables v[axes(discx), axes(discy)]))

    @test CartesianIndex(1, 10) ∈ s.Iedge
    @test all([I ∈ s.Igrid for I in s.Iedge])  
    
    @test s.Iaxies == s.Igrid
    
end

@testset "Test 02: discretization of variables, edge aligned grid" begin
    # Test centered order
    disc = MOLFiniteDifference([x=>dx, y=>dy], t; centered_order=order, grid_align=edge_align)

    s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, nottime, disc)

    discx = (x_min-dx/2): dx : (x_max+dx/2)
    discy = (y_min-dy/2): dy : (y_max+dy/2)

    @test s.grid[x] == discx
    @test s.grid[y] == discy

    @test s.axies[x] != s.grid[x]
    @test s.axies[y] != s.grid[y]

    @test s.discvars[u] == collect(first(@variables u[axes(discx), axes(discy)]))
    @test s.discvars[v] == collect(first(@variables v[axes(discx), axes(discy)]))

    @test CartesianIndex(1, 10) ∈ s.Iedge
    @test all([I ∈ s.Igrid for I in s.Iedge])  
    
    @test s.Iaxies != s.Igrid

end
