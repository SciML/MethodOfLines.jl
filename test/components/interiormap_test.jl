using MethodOfLines, ModelingToolkit, DomainSets
using Combinatorics: permutations


@testset "Test 00: recognize relevant variable for equations, time defined" begin 
    @parameters x, t 
    @variables u(..), v(..), w(..)

    Dx(d) = Differential(x)^d

    Dt = Differential(t)

    t_min= 0.
    t_max = 2.0
    x_min = 0.
    x_max = 20.0 

    dx = 1.0

    domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]


    pde = [Dt(u(t,x)) ~ +Dx(1)(u(t,x))+w(t,x),
           Dt(v(t,x)) ~ -Dx(2)(v(t,x))+w(t,x),
           Dt(w(t,x)) ~ -Dx(3)(w(t,x))+u(t,x)+v(t,x)]
    bcs = [u(0,x) ~ 0, u(t,0) ~ 0, u(t,Float64(π)) ~ 0]

    @named pdesys = PDESystem(pde,bcs,domains,[t,x],[u(t,x), v(t,x), w(t,x)])
    
    # Test centered order 
    disc = MOLFiniteDifference([x=>dx], t)
    
    depvar_ops = map(x->operation(x.val),pdesys.depvars)
    depvars = MethodOfLines.get_all_depvars(pdesys, depvar_ops)
    indvars = MethodOfLines.remove(collect(reduce(union,filter(xs->!isequal(xs, [t]), map(arguments, depvars)))),t)

    s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, disc)

    m = MethodOfLines.buildmatrix(pde, s)
    if VERSION >= v"1.7"
        @test m == hcat([1,0,0],[0,1,0],[0,0,1]) # Test the matrix is the identity matrix
    else
        @test m == [0 1 0; 0 0 1; 1 0 0] 
    end
end

@testset "Test 00a: recognize relevant variable for equations, time undefined, multiple choices" begin 
    @parameters x, t 
    @variables u(..), v(..), w(..)

    Dx(d) = Differential(x)^d

    Dt = Differential(t)

    t_min= 0.
    t_max = 2.0
    x_min = 0.
    x_max = 20.0 

    dx = 1.0

    domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]


    pde = [Dt(u(t,x)) ~ +Dx(1)(u(t,x))+Dx(1)(w(t,x)),
           Dt(v(t,x)) ~ -Dx(2)(v(t,x))+Dx(2)(w(t,x)),
           Dt(w(t,x)) ~ -Dx(3)(w(t,x))+Dx(3)(u(t,x))+Dx(3)(v(t,x))]
    bcs = [u(0,x) ~ 0, u(t,0) ~ 0, u(t,Float64(π)) ~ 0]

    @named pdesys = PDESystem(pde,bcs,domains,[t,x],[u(t,x), v(t,x), w(t,x)])

    # Test centered order 
    disc = MOLFiniteDifference([x=>dx, t=>dx])

    depvar_ops = map(x->operation(x.val),pdesys.depvars)
    depvars = MethodOfLines.get_all_depvars(pdesys, depvar_ops)
    indvars = collect(reduce(union,filter(xs->!isequal(xs, [t]), map(arguments, depvars))))

     s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, disc)

    m = MethodOfLines.buildmatrix(pde, s)

    if VERSION >= v"1.7"
        @test m == hcat([1,0,1],[0,1,1],[1,1,1]) # Test the matrix is the identity matrix
    else
        @test m == [1 1 0; 1 0 1; 1 1 1]
    end

end
#
@testset "Test 00b: recognize relevant variable for equations, time undefined, mixed derivatives, multiple choices" begin 
    @parameters x, t 
    @variables u(..), v(..), w(..)

    Dx(d) = Differential(x)^d

    Dt = Differential(t)

    t_min= 0.
    t_max = 2.0
    x_min = 0.
    x_max = 20.0 

    dx = 1.0

    domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]


    pde = [Dt(u(t,x)) ~ +Dx(1)(u(t,x))+w(t,x),
           Dt(v(t,x)) ~ -Dx(2)(v(t,x))+Dx(2)(w(t,x)),
           Dt(w(t,x)) ~ -Dx(3)(w(t,x))+Dx(3)(u(t,x)+v(t,x))]
    bcs = [u(0,x) ~ 0, u(t,0) ~ 0, u(t,Float64(π)) ~ 0]

    @named pdesys = PDESystem(pde,bcs,domains,[t,x],[u(t,x), v(t,x), w(t,x)])

    # Test centered order 
    disc = MOLFiniteDifference([x=>dx, t=>1.0])

    depvar_ops = map(x->operation(x.val),pdesys.depvars)
    depvars = MethodOfLines.get_all_depvars(pdesys, depvar_ops)
    indvars = collect(reduce(union,filter(xs->!isequal(xs, [t]), map(arguments, depvars))))

    s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, disc)

    m = MethodOfLines.buildmatrix(pde, s)

    if VERSION >= v"1.7"
        @test m == hcat([1,0,1],[0,1,1],[0,1,1]) # Test the matrix is correct
    else
        @test m == [0 1 0; 1 0 1; 1 1 1]
    end

end

@testset "Test 01a: Build variable mapping - one right choice simple" begin
    m = hcat([0, 1, 0],
             [0, 0, 1],
             [1, 0, 0])
    pdes = ["a", "b", "c"]
    vars = ["x", "y", "z"]

    @test Dict(MethodOfLines.build_variable_mapping(m, vars, pdes)) == Dict([
        "a" => "z",
        "b" => "x",
        "c" => "y"
    ])
end

@testset "Test 01b: Build variable mapping - one right choice complex" begin
    m = hcat([0, 1, 1], 
             [1, 0, 1],
             [0, 1, 0])
    pdes = ["a", "b", "c"]
    vars = ["x", "y", "z"]

    @test Dict(MethodOfLines.build_variable_mapping(m, vars, pdes)) == Dict([
        "a" => "y",
        "b" => "z",
        "c" => "x"
    ])
end

@testset "Test 01c: Build variable mapping - two right choices" begin
    m = hcat([0, 1, 1], 
             [1, 0, 1],
             [1, 1, 0])
    pdes = ["a", "b", "c"]
    vars = ["x", "y", "z"]

    @test Dict(MethodOfLines.build_variable_mapping(m, vars, pdes)) == Dict([
        "a" => "y",
        "b" => "z",
        "c" => "x"
    ]) || Dict(MethodOfLines.build_variable_mapping(m, pdes, vars)) == Dict([
        "a" => "z",
        "b" => "x",
        "c" => "y"
    ])
end

@testset "Test 01d: Build variable mapping - any choice correct" begin
    m = hcat([1, 1, 1], 
             [1, 1, 1],
             [1, 1, 1])
    pdes = ["a", "b", "c"]
    vars = ["x", "y", "z"]

    perms = permutations(vars)
    dict = Dict(MethodOfLines.build_variable_mapping(m, vars, pdes))
    @test any(map(perm -> dict == Dict(["a" => perm[1],
                                        "b" => perm[2],
                                        "c" => perm[3]]), perms))
end

@testset "Test 01e: Build variable mapping - no correct mapping" begin
    m = hcat([1, 1, 0], 
             [1, 1, 0],
             [1, 1, 0])

    pdes = ["a", "b", "c"]
    vars = ["x", "y", "z"]
    try
        MethodOfLines.build_variable_mapping(m, vars, pdes)
        @test false
    catch e
        @test e isa AssertionError
    end
end