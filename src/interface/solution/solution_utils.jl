# TODO: Check if the grid is uniform and use the supported higher order interpolations

_without(xs::Tuple, i) = ntuple(j -> xs[j < i ? j : j + 1], length(xs) - 1)

# The interpolation of a solution saved at a single time (for example with `save_on = false`):
# `spatial` interpolates the saved slice, and the time argument at position `ti` must be that time.
struct SingleTimeInterpolation{I, T}
    spatial::I
    t::T
    ti::Int
end

struct TimeOnlyValue{T}
    value::T
end

(f::TimeOnlyValue)() = f.value

function (f::SingleTimeInterpolation)(args::Vararg{Union{Number, AbstractVector}})
    targ = args[f.ti]
    all(==(f.t), targ) || throw(
        ArgumentError(
            "The solution was saved only at t = $(f.t), so it cannot be interpolated at t = $(targ)."
        )
    )
    values = f.spatial(_without(args, f.ti)...)
    targ isa Number && return values
    slice = values isa Number ? fill(values) : values
    d = min(f.ti, ndims(slice) + 1)
    return repeat(
        reshape(slice, ntuple(j -> j == d ? 1 : size(slice, j < d ? j : j - 1), ndims(slice) + 1)),
        outer = ntuple(j -> j == d ? length(targ) : 1, ndims(slice) + 1)
    )
end

function build_interpolation(
        umap, dvs, ivs, ivgrid, sol, pdesys, replaced_vars, discretespace
    )
    return mapreduce(vcat, Num.(dvs)) do k
        args = arguments(k.val)
        nodes = (
            map(args) do arg
                i = findfirst(isequal(arg), ivs)
                @assert i !== nothing "Independent variable $arg not found in ivs $ivs."
                isequal(arg, discretespace.time) ? collect(ivgrid[i]) :
                    collect(ivgrid[i]) .+ staggered_shift(discretespace, k.val, arg)
            end...,
        )
        @assert all(
            map(
                pair -> isequal(first(pair[1]), pair[2]), zip(axes.(nodes), axes(umap[k]))
            )
        ) "Please ensure that the order that your independent variables appear in your PDESystem matches the order that they appear in the argument signature of your dependent variables. For example, if you have a PDESystem with independent variables `[t, x, y]`, and a dependent variable `u(t, x, y)`, then you should use `u(t, x, y)` instead of `u(t, y, x)` in your sytstem definition.\nFound a problem with the dependent variable $k, given that your ivs are $(get_ivs(pdesys))."
        kreplaced = get(replaced_vars, k, nothing)
        # staggered points start half a step inside the domain edge, so evaluating on
        # the grid (`sol(t, :)`) reaches half a step past the first point
        shifted = arg -> !isequal(arg, discretespace.time) &&
            !iszero(staggered_shift(discretespace, k.val, arg))
        gridded(nodes, data) = begin
            interp = interpolate(nodes, data, Gridded(Linear()))
            any(shifted, args) ? Interpolations.extrapolate(interp, Interpolations.Line()) : interp
        end
        ti = findfirst(isequal(discretespace.time), args)
        if all(length.(nodes) .> 1)
            interp = gridded(nodes, umap[k])
        elseif ti !== nothing && length(nodes[ti]) == 1 && all(length.(_without(nodes, ti)) .> 1)
            # a field of time only has no spatial interpolation to build
            spatial = length(nodes) == 1 ? TimeOnlyValue(only(umap[k])) :
                gridded(_without(nodes, ti), dropdims(umap[k]; dims = ti))
            interp = SingleTimeInterpolation(spatial, only(nodes[ti]), ti)
        else
            i = findfirst(a -> a <= 1, length.(nodes))
            @warn "Solution has length 1 in dimension $(ivs[i]). Interpolation will not be possible for variable $k. Solution return code is $(sol.retcode)."
            interp = nothing
        end
        if isnothing(kreplaced)
            [k => interp]
        else
            [k => interp, kreplaced => interp]
        end
    end |> Dict
end

# function build_interpolation(umap, dvs, ivs, ivgrid, sol{G}, pdesys, replaced_vars) where {G<:StaggeredGrid}
# end

sym_to_index(sym, syms) = findfirst(isequal(sym), syms)

# The PDE solution constructors extract states element by element and evaluate observed
# functions on them, which scalar-indexes device-resident states (e.g. `CuArray`s). Copy
# those to the host once up front; host states are returned as is.
function host_solution(sol::SciMLBase.AbstractODESolution)
    eltype(sol.u) <: Array && return sol
    return SciMLBase.build_solution(
        sol.prob, sol.alg, sol.t, Array.(sol.u);
        retcode = sol.retcode, calculate_error = false
    )
end

function host_solution(sol::SciMLBase.NonlinearSolution)
    sol.u isa Array && return sol
    return SciMLBase.build_solution(
        sol.prob, sol.alg, Array(sol.u), sol.resid;
        retcode = sol.retcode, calculate_error = false
    )
end

# The state index of each element of `discu`, or `nothing` where it is observed.
function grid_indices(odesys, discu)
    return map(I -> SymbolicIndexingInterface.variable_index(odesys, discu[I]), CartesianIndices(discu))
end

# The values of the observed elements `obs` of `discu`, from a single observed function
# rather than one generated and compiled per element.
function grid_observed_values(sol, discu, obs)
    return SymbolicIndexingInterface.getsym(sol, [safe_unwrap(discu[I]) for I in obs])(sol)
end

iscomplex(A::SciMLBase.AbstractPDESolution) = !isnothing(A.disc_data.complexmap)
