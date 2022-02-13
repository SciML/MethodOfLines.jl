struct InteriorMap
    var
    I
end

function InteriorMap(pdes, boundarymap, s::DiscreteSpace{N, M}) where {N,M}
    @assert length(pdes) == M "There must be the same number of equations and unknowns, got $(length(pdes)) equations and $(M) unknowns"
    m = buildmatrix(pdes, s)
    varmap = build_variable_mapping(m, s.ū, pdes)

    # Determine the interiors for each pde
    interior = map(pdes) do pde
        u = varmap[pde]
        boundaries = boundarymap[operation(u)]
        lower = zeros(Int, N)
        upper = zeros(Int, N)
        # Determine thec number of points to remove from each end of the domain for each dimension
        for b in boundaries
            clip_interior!!(lower, upper, b, s.x2i)
        end           
        args = remove(arguments(u), s.time)
        pde => s.Igrid[u][[(1 + lower[s.x2i[x]] : length(s.grid[x]) - upper[s.x2i[x]]) for x in args]]
    end
    return InteriorMap(Dict(varmap), Dict(interior))
end


function buildmatrix(pdes, s::DiscreteSpace{N,M}) where {N,M}
    m = zeros(Int, M, M)
    elegiblevars = [getvars(pde, s) for pde in pdes]
    u2i = Dict([u => k for (k, u) in enumerate(s.ū)])
    for (i, vars) in enumerate(elegiblevars)
        for var in vars
            m[i, u2i[var]] = 1
        end
    end
    return m
end

function build_variable_mapping(m, vars, pdes)
    varpdemap = []
    rows = sum(m, dims=2)
    cols = sum(m, dims=1)
    i = findfirst(isequal(0), rows)
    j = findfirst(isequal(0), cols)
    @assert i !== nothing "Equation $(pde[i]) is not an equation for any of the dependent variables."
    @assert j !== nothing "Variable $(vars[j]) does not appear in any equation, therefore cannot be solved for"
    for k in 1:N
        # Check if any of the pdes only have one valid variable
        cols = sum(m, dims=1)
        j = findfirst(isequal(1), cols)#
        if j !== nothing 
            for i in 1:N
                if m[i, j] == 1
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] = 0
                    m[:, j] = 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables only have one valid pde
        rows = sum(m, dims=2)
        i = findfirst(isequal(1), rows)
        if i !== nothing
            for j in 1:N
                if m[i, j] == 1
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] = 0
                    m[:, j] = 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables have more than one valid pde, and pick one
        i = findfirst(x -> x > 1, rows)
        if i !== nothing 
            for j in 1:N
                if m[i, j] == 1
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] = 0
                    m[:, j] = 0
                    break
                end
            end
            continue
        end
    end
    @assert length(varpdemap) == N "Could not map all variables to pdes"
    return varpdemap
end
    """
Counts the Differential operators for given variable x, and the derived variables at that order
"""
function get_order_and_depvars(term, x::Symbolics.Symbolic, s)
    S = Symbolics
    SU = SymbolicUtils
    if term ∈ s.ū
        return (0, [term])
    else
        op = SU.operation(term)
        children = map(arg -> get_order_and_depvar(arg, x, s), SU.arguments(term))
        count = [child[1] for child in children]
        # if there are more than one depvars at that order, include all of them
        is = findall(a -> a == max(count), count)
        vars = unique(reduce(vcat, [child[2] for child in children[is]]))
        
        if op isa Differential && op.x === x
            return (1 + sum(count), vars) 
        end
        return (sum(count), vars)
    end
end

function getvars(pde, s)
    if s.time !== nothing
        ct, ut = get_order_and_depvars(pde, s.time, s)
        if ct > 0
            return ut
        end
    end
    countsx = map(s.x̄) do x
        get_order_and_depvars(pde, x, s)
    end
    counts = [c[1] for c in countsx]
    is = findall(a -> a == max(count), count)
    return reduce(vcat, [c[2] for c in countsx[is]])
end

