# Trapezoidal integrals. The pointwise path expands a running sum at each index
# (`_euler_integral`); the array path writes the same increments once as a scan.

"""
    axis_cumsum_pad(A, dim)

`cumsum` along `dim` after a leading zero slab. The result is one longer than
`A` and starts at 0. Per-axis wrappers exist because `@register_array_symbolic`
evaluates `size` with a symbolic `dim`.
"""
function _axis_cumsum_pad(A::AbstractArray, dim::Integer)
    dim = Int(dim)
    sz = ntuple(i -> i == dim ? size(A, i) + 1 : size(A, i), ndims(A))
    padded = similar(A, sz)
    selectdim(padded, dim, 1) .= zero(eltype(A))
    copyto!(selectdim(padded, dim, 2:size(padded, dim)), A)
    return cumsum(padded; dims = dim)
end

axis_cumsum_pad_1(A::AbstractArray) = _axis_cumsum_pad(A, 1)
axis_cumsum_pad_2(A::AbstractArray) = _axis_cumsum_pad(A, 2)
axis_cumsum_pad_3(A::AbstractArray) = _axis_cumsum_pad(A, 3)

Symbolics.@register_array_symbolic axis_cumsum_pad_1(A::AbstractArray) begin
    size = ntuple(i -> i == 1 ? size(A, i) + 1 : size(A, i), ndims(A))
    eltype = Real
end
Symbolics.@register_array_symbolic axis_cumsum_pad_2(A::AbstractArray) begin
    size = ntuple(i -> i == 2 ? size(A, i) + 1 : size(A, i), ndims(A))
    eltype = Real
end
Symbolics.@register_array_symbolic axis_cumsum_pad_3(A::AbstractArray) begin
    size = ntuple(i -> i == 3 ? size(A, i) + 1 : size(A, i), ndims(A))
    eltype = Real
end

function axis_cumsum_pad(A, dim::Integer)
    dim == 1 && return axis_cumsum_pad_1(A)
    dim == 2 && return axis_cumsum_pad_2(A)
    dim == 3 && return axis_cumsum_pad_3(A)
    throw(ArrayFormFallback("integral scan pad is registered for axes 1:3, got $dim"))
end

"""
    axis_sum(A, dim)

`dropdims(sum(A; dims = dim); dims = dim)`. Base `sum(; dims =)` on a
MethodOfLines slice can leave a broken size.
"""
_axis_sum(A::AbstractArray, dim::Integer) = dropdims(sum(A; dims = Int(dim)); dims = Int(dim))
axis_sum_1(A::AbstractArray) = _axis_sum(A, 1)
axis_sum_2(A::AbstractArray) = _axis_sum(A, 2)
axis_sum_3(A::AbstractArray) = _axis_sum(A, 3)

Symbolics.@register_array_symbolic axis_sum_1(A::AbstractArray) begin
    size = ntuple(i -> size(A, i + 1), ndims(A) - 1)
    eltype = Real
end
Symbolics.@register_array_symbolic axis_sum_2(A::AbstractArray) begin
    size = ntuple(i -> size(A, i < 2 ? i : i + 1), ndims(A) - 1)
    eltype = Real
end
Symbolics.@register_array_symbolic axis_sum_3(A::AbstractArray) begin
    size = ntuple(i -> size(A, i < 3 ? i : i + 1), ndims(A) - 1)
    eltype = Real
end

function axis_sum(A, dim::Integer)
    dim == 1 && return axis_sum_1(A)
    dim == 2 && return axis_sum_2(A)
    dim == 3 && return axis_sum_3(A)
    throw(ArrayFormFallback("integral axis sum is registered for axes 1:3, got $dim"))
end

trapezoid_gap_weights(dx::Number, n::Integer) = fill(dx / 2, max(n - 1, 0))
function trapezoid_gap_weights(dx::AbstractVector, n::Integer)
    return collect(dx[1:max(n - 1, 0)] ./ 2)
end

function trapezoid_node_weights(dx::Number, n::Integer)
    n <= 1 && return zeros(typeof(dx / 2), n)
    w = fill(dx, n)
    w[1] = dx / 2
    w[n] = dx / 2
    return w
end
function trapezoid_node_weights(dx::AbstractVector, n::Integer)
    w = zeros(eltype(dx) <: Integer ? Float64 : eltype(dx), n)
    n <= 1 && return w
    w[1] = dx[1] / 2
    w[n] = dx[n - 1] / 2
    @inbounds for k in 2:(n - 1)
        w[k] = (dx[k - 1] + dx[k]) / 2
    end
    return w
end

function reshape_along(vals, j, N)
    N == 1 && return vals
    return reshape(vals, ntuple(i -> i == j ? length(vals) : 1, N))
end

# A registered scale keeps a length-n weight vector as one argument.
array_weight_scale(w::AbstractArray, A::AbstractArray) = broadcast(*, w, A)
Symbolics.@register_array_symbolic array_weight_scale(w::AbstractArray, A::AbstractArray) begin
    size = size(A)
    eltype = Real
end

function scale_by_weights(w, A)
    w isa Number && return broadcast(*, w, A)
    return array_weight_scale(w, A)
end

function integral_term(s, u, x, whole::Bool)
    lo, hi = s.vars.intervals[x]
    return Integral(x in DomainSets.ClosedInterval(lo, whole ? hi : Num(x)))(u)
end

function array_has_integral(pde, u, x, s)
    u = depvar(u, s)
    function has(expr)
        expr = safe_unwrap(expr)
        iscall(expr) || return false
        op = operation(expr)
        if op isa Integral && isequal(op.domain.variables, x)
            return isequal(depvar(only(arguments(expr)), s), u)
        end
        return any(has, arguments(expr))
    end
    return has(pde.lhs) || has(pde.rhs)
end

_iv_in(x, xs) = any(y -> isequal(x, y), xs)
function array_same_ivs(a, b)
    length(a) == length(b) || return false
    return all(x -> _iv_in(x, b), a) && all(y -> _iv_in(y, a), b)
end

function array_compatible_depvar(u, args, pde, s)
    uivs = ivs(u, s)
    array_same_ivs(uivs, args) && return true
    isempty(uivs) && return true
    extra = [y for y in uivs if !_iv_in(y, args)]
    missing = [y for y in args if !_iv_in(y, uivs)]
    isempty(missing) || return false
    return !isempty(extra) && all(y -> array_has_integral(pde, u, y, s), extra)
end

function array_scalar_discvar(u, s)
    disc = s.discvars[depvar(u, s)]
    return ndims(disc) == 0 ? disc[] : only(disc)
end

"""
Index ranges of `u` for an integral along `x`: the full axis in `x`, the
equation core (or the full axis) in every other independent variable.
"""
function array_integral_ranges(u, s, x, ranges, indexmap; xspan = nothing)
    u = depvar(u, s)
    return map(ivs(u, s)) do y
        if isequal(y, x)
            return xspan === nothing ? (1:length(s, x)) : xspan
        elseif haskey(indexmap, y)
            return ranges[indexmap[y]]
        else
            return 1:length(s, y)
        end
    end
end

function array_integral_axis_index(u, s, x)
    uivs = ivs(depvar(u, s), s)
    j = findfirst(y -> isequal(y, x), uivs)
    j === nothing && throw(ArrayFormFallback("integral axis $x is not an argument of $u"))
    return j, length(uivs)
end

"""
Trapezoidal running integral of `u` along `x` as a slice over the core
described by `ranges`. Reads the full axis of `u` in `x`.
"""
function array_cumulative_integral(u, s, x, ranges, indexmap)
    u = depvar(u, s)
    arr = array_variable(u, s)
    j, N = array_integral_axis_index(u, s, x)
    n = length(s, x)
    rs = array_integral_ranges(u, s, x, ranges, indexmap)
    n <= 1 && return arr[rs...] .* 0
    dx = s.dxs[x]
    w = dx isa Number ? dx / 2 : reshape_along(trapezoid_gap_weights(dx, n), j, N)
    rs_lo = ntuple(i -> i == j ? (1:(n - 1)) : rs[i], N)
    rs_hi = ntuple(i -> i == j ? (2:n) : rs[i], N)
    inc = scale_by_weights(w, broadcast(+, arr[rs_lo...], arr[rs_hi...]))
    I_full = axis_cumsum_pad(inc, j)
    xspan = haskey(indexmap, x) ? ranges[indexmap[x]] : (1:n)
    rs_out = ntuple(i -> i == j ? xspan : Colon(), N)
    return I_full[rs_out...]
end

"""
Whole-domain trapezoidal integral of `u` along `x`. Rank drops in `x`: a 1D
integrand becomes a scalar; a higher-dimensional integrand keeps its other axes.
"""
function array_whole_domain_integral(u, s, x, ranges, indexmap)
    u = depvar(u, s)
    arr = array_variable(u, s)
    j, N = array_integral_axis_index(u, s, x)
    n = length(s, x)
    rs = array_integral_ranges(u, s, x, ranges, indexmap)
    u_full = arr[rs...]
    w = reshape_along(trapezoid_node_weights(s.dxs[x], n), j, N)
    wu = scale_by_weights(w, u_full)
    N == 1 && return sum(wu)
    return axis_sum(wu, j)
end

"""
Whole-domain trapezoidal integral as a scalar at the transverse location of `II`.
"""
function compact_whole_domain_integral(s, u, x, II, indexmap)
    u = depvar(u, s)
    arr = array_variable(u, s)
    j, N = array_integral_axis_index(u, s, x)
    n = length(s, x)
    IIu = wd_integral_Idx(II, s, u, x, indexmap)
    rs = ntuple(N) do i
        i == j ? (1:n) : IIu[i]:IIu[i]
    end
    w = reshape_along(trapezoid_node_weights(s.dxs[x], n), j, N)
    return sum(scale_by_weights(w, arr[rs...]))
end

function array_integral_rules(
        s, depvars, ranges, indexmap;
        bvar = nothing, staggered = false
    )
    rules = Pair[]
    for u in depvars
        u = depvar(u, s)
        for x in ivs(u, s)
            if !staggered
                push!(
                    rules,
                    safe_unwrap(integral_term(s, u, x, false)) =>
                        array_cumulative_integral(u, s, x, ranges, indexmap)
                )
            end
            if !haskey(indexmap, x) || isequal(x, bvar)
                push!(
                    rules,
                    safe_unwrap(integral_term(s, u, x, true)) =>
                        array_whole_domain_integral(u, s, x, ranges, indexmap)
                )
            end
        end
    end
    return rules
end

# use the trapezoid rule
function _euler_integral(II, s, jx, u, ufunc, dx::Number) #where {T,N,Wind,DX<:Number}
    j, x = jx
    if II[j] == 1
        return Num(0)
    end
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication
    Itap = [II - I1, II]
    weights = [dx / 2, dx / 2]

    return sym_dot(weights, ufunc(u, Itap, x)) +
        _euler_integral(II - I1, s, jx, u, ufunc, dx)
end

# Nonuniform dx
function _euler_integral(II, s, jx, u, ufunc, dx::AbstractVector) #where {T,N,Wind,DX<:Number}
    j, x = jx
    if II[j] == 1
        return Num(0)
    end
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication
    Itap = [II - I1, II]
    weights = fill(dx[II[j] - 1] / 2, 2)

    return sym_dot(weights, ufunc(u, Itap, x)) +
        _euler_integral(II - I1, s, jx, u, ufunc, dx)
end

function euler_integral(II, s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    return _euler_integral(II, s, jx, u, ufunc, dx)
end

# An integral across the whole domain (xmin .. xmax)
function whole_domain_integral(II, s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    if II[j] == 1 && length(s, x) == 1
        return Num(0)
    end
    if II[j] == length(s, x)
        return _euler_integral(II, s, jx, u, ufunc, dx)
    end

    dist2max = length(s, x) - II[j]
    I1 = unitindex(ndims(u, s), j)
    Imax = II + dist2max * I1
    return _euler_integral(Imax, s, jx, u, ufunc, dx)
end

@inline function generate_euler_integration_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms
    )
    ufunc(u, I, x) = s.discvars[u][I]

    eulerrules = reduce(
        safe_vcat,
        [
            [
                    Integral(
                        x in DomainSets.ClosedInterval(
                            s.vars.intervals[x][1],
                            Num(x)
                        )
                    )(u) => euler_integral(
                        Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc
                    )
                    for x in ivs(u, s)
                ]
                for u in depvars
        ],
        init = []
    )
    return eulerrules
end

function wd_integral_Idx(II::CartesianIndex, s::DiscreteSpace, u, x, indexmap)
    # We need to construct a new index as indices may be of different size
    length(ivs(u, s)) == 0 && return CartesianIndex()
    # A hack using the boundary value re-indexing function to get an index that will work
    u_ = pde_substitute(u, Dict(x => s.axies[x][end]))
    II = newindex(u_, II, s, indexmap)
    return II
end

@inline function generate_whole_domain_integration_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms, bvar = nothing
    )
    ufunc(u, I, x) = s.discvars[u][I]
    wholedomainrules = reduce(
        safe_vcat,
        [
            [
                    Integral(
                        x in DomainSets.ClosedInterval(
                            s.vars.intervals[x][1],
                            s.vars.intervals[x][2]
                        )
                    )(u) => begin
                        try
                            compact_whole_domain_integral(s, u, x, II, indexmap)
                    catch e
                            e isa InterruptException && rethrow(e)
                            whole_domain_integral(
                                wd_integral_Idx(II, s, u, x, indexmap),
                                s, (x2i(s, u, x), x), u, ufunc
                            )
                    end
                    end
                    for x in filter(x -> (!haskey(indexmap, x) | isequal(x, bvar)), ivs(u, s))
                ]
                for u in depvars
        ],
        init = []
    )
    return wholedomainrules
end
