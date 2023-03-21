
"""
A function that creates a tuple of CartesianIndices of unit length and `N` dimensions, one pointing along each dimension.
"""
function unitindices(N::Int) #create unit CartesianIndex for each dimension
    null = zeros(Int, N)
    if N == 0
        return CartesianIndex()
    else
        return map(1:N) do i
            unit_i = copy(null)
            unit_i[i] = 1
            CartesianIndex(Tuple(unit_i))
        end
    end
end

"""
    unitindex(N, j)
Get a unit `CartesianIndex` in dimension `j` of length `N`.
"""
unitindex(N, j) = CartesianIndex(ntuple(i -> i == j, N))

@inline clip(II::CartesianIndex{M}, j, N) where {M} = II[j] > N ? II - unitindices(M)[j] : II

remove(args, t) = filter(x -> t === nothing || !isequal(safe_unwrap(x), safe_unwrap(t)), args)
remove(v::AbstractVector, a::Number) = filter(x -> !isequal(x, a), v)


half_range(x) = -div(x, 2):div(x, 2)

d_orders(x, pdeeqs) = reverse(sort(collect(union((differential_order(pde.rhs, safe_unwrap(x)) for pde in pdeeqs)..., (differential_order(pde.lhs, safe_unwrap(x)) for pde in pdeeqs)...))))

insert(args...) = insert!(args[1], args[2:end]...)

####
# Utils for DerivativeOperator generation in schemes
####

index(i::Int, N::Int) = i + div(N, 2) + 1

function generate_coordinates(i::Int, stencil_x, dummy_x,
    dx::AbstractVector{T}) where {T<:Real}
    len = length(stencil_x)
    stencil_x .= stencil_x .* zero(T)
    for idx in 1:div(len, 2)
        shifted_idx1 = index(idx, len)
        shifted_idx2 = index(-idx, len)
        stencil_x[shifted_idx1] = stencil_x[shifted_idx1-1] + dx[i+idx-1]
        stencil_x[shifted_idx2] = stencil_x[shifted_idx2+1] - dx[i-idx]
    end
    return stencil_x
end

function _get_gridloc(s, ut, is...)
    u = Sym{SymbolicUtils.FnType{Tuple, Real}}(nameof(operation(ut)))
    u = operation(s.ū[findfirst(isequal(u), operation.(s.ū))])
    args = remove(s.args[u], s.time)
    gridloc = map(enumerate(args)) do (i, x)
        s.grid[x][is[i]]
    end
    return (u, gridloc)
end

function get_gridloc(u, s)
    if isequal(operation(u), getindex)
        # Remember arguments of getindex have u(t) first
        return _get_gridloc(s, arguments(u)...)
    else
        return (operation(u), [])
    end
end

function generate_function_from_gridlocs(analyticmap, gridlocs, s)
    is_t_first_map = Dict(map(s.ū) do u
        operation(u) => (findfirst(x -> isequal(s.time, x), arguments(u)) == 1)
    end)

    opsmap = Dict(map(s.ū) do u
        operation(u) => u
    end)

    fs_ = map(gridlocs) do (uop, x̄)
        is_t_first = is_t_first_map[uop]
        _f = analyticmap[opsmap[uop]]
        if is_t_first
            return (p, t) -> _f(p, t, x̄...)
        else
            return (p, t) -> _f(p, x̄..., t)
        end
    end

    f = (u0, p, t) -> map(fs_) do f_
        f_(p, t)
    end

    return f
end

function newindex(u_, II, s, indexmap)
    u = depvar(u_, s)
    args_ = remove(arguments(u_), s.time)
    args = ivs(u, s)
    is = map(enumerate(args_)) do (j, x)
        if haskey(indexmap, x)
            II[indexmap[x]]
        elseif safe_unwrap(x) isa Number
            if isequal(x, s.axies[args[j]][1])
                1
            elseif isequal(x, s.axies[args[j]][end])
                length(s, args[j])
            else
                error("Boundary value $u_ is not defined at the boundary of the domain, or problem with index adaptation, please post an issue.")
            end
        else
            error("Invalid boundary value found $u_, or problem with index adaptation, please post an issue.")
        end
    end
    II = CartesianIndex(is...)
    return II
end

@inline function safe_vcat(a, b)
    if length(a) == 0 && length(b) == 0
        return []
    elseif length(a) == 0
        return b
    elseif length(b) == 0
        return a
    else
        return vcat(a, b)
    end
end

function chebyspace(N, dom)
    interval = dom.domain
    a, b = DomainSets.infimum(interval), DomainSets.supremum(interval)
    x = reverse([(a + b) / 2 + (b - a) / 2 * cospi((2k - 1) / (2N)) for k in 1:N])
    x[1] = a
    x[end] = b
    return dom.variables => x
end

@inline function sym_dot(a, b)
    mapreduce((+), zip(a, b)) do (a_, b_)
        a_ * b_
    end
end
