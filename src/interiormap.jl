struct InteriorMap
    var
    pde
    I
    lower
    upper
end

function InteriorMap(pdes, boundarymap, s::DiscreteSpace{N, M}) where {N,M}
    @assert length(pdes) == M "There must be the same number of equations and unknowns, got $(length(pdes)) equations and $(M) unknowns"
    m = buildmatrix(pdes, s)
    varmap = build_variable_mapping(m, s.ū, pdes)

    # Determine the interiors for each pde
    vlower = []
    vupper = []

    interior = map(enumerate(pdes)) do (i,pde)
        u = varmap[i].second
        boundaries = boundarymap[operation(u)]
        n = ndims(u, s)
        lower = zeros(Int, n)
        upper = zeros(Int, n)
        # Determine thec number of points to remove from each end of the domain for each dimension
        for b in boundaries
            clip_interior!!(lower, upper, s, b)
        end           
        push!(vlower, pde => lower)
        push!(vupper, pde => upper)
        args = remove(arguments(u), s.time)
        # Don't update this x2i, it is correct.
        pde => s.Igrid[u][[(1 + lower[x2i(s, u, x)] : length(s.grid[x]) - upper[x2i(s, u, x)]) for x in args]...]
    end
    pdemap = [k.second => k.first for k in varmap]
    return InteriorMap(Dict(varmap), Dict(pdemap), Dict(interior), Dict(vlower), Dict(vupper))
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
    N = length(pdes)
    rows = sum(m, dims=2)
    cols = sum(m, dims=1)
    i = findfirst(isequal(0), rows)
    j = findfirst(isequal(0), cols)
    @assert i === nothing "Equation $(pdes[i[1]]) is not an equation for any of the dependent variables."
    @assert j === nothing "Variable $(vars[j[2]]) does not appear in any equation, therefore cannot be solved for"
    for k in 1:N
        # Check if any of the pdes only have one valid variable
        cols = sum(m, dims=1)
        j = findfirst(isequal(1), cols)
        if j !== nothing 
            j = j[2]
            for i in 1:N
                if m[i, j] == 1
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] .= 0
                    m[:, j] .= 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables only have one valid pde
        rows = sum(m, dims=2)
        i = findfirst(isequal(1), rows)
        if i !== nothing
            i = i[1]
            for j in 1:N
                if m[i, j] == 1
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] .= 0
                    m[:, j] .= 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables have more than one valid pde, and pick one
        i = findfirst(x -> x > 1, rows)
        if i !== nothing 
            i = i[1]
            for j in 1:N
                if m[i, j] == 1
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] .= 0
                    m[:, j] .= 0
                    break
                end
            end
            continue
        end
    end
    @assert length(varpdemap) == N "Could not map all variables to pdes"
    return varpdemap
end

@inline function split(children) 
    count = [child[1] for child in children]
    # if there are more than one depvars at that order, include all of them
    is = findall(a -> a == maximum(count), count)
    vars = unique(reduce(vcat, [child[2] for child in children[is]]))
    return (sum(count), vars)
end
"""
Counts the Differential operators for given variable x, and the derived variables at that order
"""
function get_order_and_depvars(term, x, s)
    if !istree(term)
        return (0, [])
    end
    S = Symbolics
    SU = SymbolicUtils
    #@show term
    if findfirst(isequal(term), s.ū) !== nothing
        return (0, [term])
    else
        op = SU.operation(term)
        children = map(arg -> get_order_and_depvars(arg, x, s), SU.arguments(term))
        count, vars = split(children)
        if op isa Differential && op.x === x
            return (1 + count, vars) 
        end
        return (count, vars)
    end
end

get_order_and_depvars(eq::Equation, x, s) = split([get_order_and_depvars(eq.lhs, x, s), get_order_and_depvars(eq.rhs, x, s)])

function getvars(pde, s)
    ct = 0
    ut = []
    if s.time !== nothing
        ct, ut = get_order_and_depvars(pde, s.time, s)
        if ct > 0
            return ut
        end
    end
    countsx = vcat(map(s.x̄) do x
        get_order_and_depvars(pde, x, s)
    end, (ct, ut))
    _, vars = split(countsx)
    return vars
end

