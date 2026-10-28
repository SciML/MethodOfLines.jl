# PointPort lowering. A port selects nodes `I` with weights `w_I` (a partition of unity)
# per spatial direction and takes their tensor product. The across variable is
# `Σ w_I u_I`; a flow `Q` enters node `I` as the source density `scale * Q * w_I / V_I`.
# One weight table serves both directions, so the coupling is conservative
# (`Σ q_I V_I = Q`) and passive (`Σ q_I V_I u_I = Q * across`).

function port_depvar(spec::PortSpec, s::DiscreteSpace)
    i = findfirst(u -> isequal(operation(u), operation(spec.var)), s.ū)
    i === nothing &&
        throw(ArgumentError("PointPort on $(spec.var): not a dependent variable of the PDESystem."))
    return s.ū[i]
end

# Half the distance between a node's neighbours: a full cell inside, a half cell at a
# boundary end. The last node of a direction joined across its upper end (periodic or
# interface) holds a full cell; a periodic one wraps onto the first cell.
function port_control_width(grid, i; periodic = false, upper_open = false)
    n = lastindex(grid)
    i == n && periodic && return (grid[2] - grid[1] + grid[n] - grid[n - 1]) / 2
    i == n && upper_open && return grid[n] - grid[n - 1]
    return (grid[min(i + 1, n)] - grid[max(i - 1, firstindex(grid))]) / 2
end

# Rounding tolerance at grid node `i`, scaled by its nearest neighbour gap so that it stays
# below the spacing on strongly stretched grids.
node_tol(grid, i) = sqrt(eps(Float64)) * minimum(
    abs(grid[j] - grid[i]) for j in (i - 1, i + 1) if checkbounds(Bool, grid, j);
    init = Inf
)

function select_port_nodes_1d(
        sel, grid, domain, x0, allowed, x; periodic = false, upper_open = false
    )
    first(domain) - node_tol(grid, firstindex(grid)) <= x0 <=
        last(domain) + node_tol(grid, lastindex(grid)) ||
        throw(ArgumentError("PointPort coordinate $x = $x0 lies outside the domain [$(first(domain)), $(last(domain))]."))
    n = port_npoints(sel)
    idx = collect(allowed)
    if periodic
        # node 1 is the copy of node n; unroll one period on each side to wrap the stencil
        idx = filter(!=(firstindex(grid)), idx)
        k = min(n, length(idx))
        L = last(domain) - first(domain)
        xs = vcat(grid[idx[(end - k + 1):end]] .- L, grid[idx], grid[idx[1:k]] .+ L)
        idx = vcat(idx[(end - k + 1):end], idx, idx[1:k])
    else
        xs = grid[idx]
        first(xs) - node_tol(grid, first(idx)) <= x0 <= last(xs) + node_tol(grid, last(idx)) || throw(
            ArgumentError(
                "PointPort coordinate $x = $x0 carries a flow but lies outside the interior nodes " *
                    "$x ∈ [$(first(xs)), $(last(xs))] that hold the PDE; the nodes nearer the boundary " *
                    "hold boundary conditions. Put the flow in a flux boundary condition instead, " *
                    "e.g. `-k * Dx(T(t, 0)) ~ heater.Q_flow / A`."
            )
        )
    end
    # a point on a node uses that node alone; "on" is judged against the local spacing,
    # since stretched grids can hold nodes far closer together than the domain width
    near = argmin(abs.(xs .- x0))
    gaps = [abs(xs[j] - xs[near]) for j in (near - 1, near + 1) if checkbounds(Bool, xs, j)]
    if abs(xs[near] - x0) <= sqrt(eps(Float64)) * minimum(gaps; init = Inf)
        nodes = idx[near:near]
        return nodes, [1.0], port_control_width.(Ref(grid), nodes; periodic, upper_open)
    end
    n <= length(unique(idx)) ||
        throw(ArgumentError("PointPort selection $sel needs $n nodes in $x but only $(length(unique(idx))) are available."))
    lo = clamp(searchsortedlast(xs, x0), 1, length(xs) - 1)
    window = if n == 1
        inside = k -> first(domain) <= xs[k] <= last(domain)
        dlo, dhi = x0 - xs[lo], xs[lo + 1] - x0
        pick = abs(dlo - dhi) <= sqrt(eps(Float64)) * (xs[lo + 1] - xs[lo]) ?
            (inside(lo) || !inside(lo + 1) ? lo : lo + 1) : (dlo < dhi ? lo : lo + 1)
        pick:pick
    else
        # grow outward from the bracketing pair, nearer side first, so x0 stays inside
        a, b = lo, lo + 1
        while b - a + 1 < n
            left = a > 1 ? x0 - xs[a - 1] : Inf
            right = b < length(xs) ? xs[b + 1] - x0 : Inf
            left <= right ? (a -= 1) : (b += 1)
        end
        a:b
    end
    w = length(window) == 1 ? [1.0] : calculate_weights(0, x0, xs[window])
    nodes = idx[window]
    return nodes, w, port_control_width.(Ref(grid), nodes; periodic, upper_open)
end

"""
    select_port_nodes(spec, s; restrict_to = nothing, periodic = nothing, upper_open = nothing)

Nodes, weights and control volumes of the port `spec` on the grid of `s`, as vectors
`(nodes::Vector{CartesianIndex}, weights, volumes)` in the index space of the port
variable. `restrict_to` gives, per spatial direction, the index range the nodes may come
from; by default every grid node is allowed. `periodic` and `upper_open` flag, per
direction, a periodic axis and an axis joined to another domain at its upper end.
"""
function select_port_nodes(
        spec::PortSpec, s::DiscreteSpace; restrict_to = nothing,
        periodic = nothing, upper_open = nothing
    )
    u = port_depvar(spec, s)
    xs = ivs(u, s)
    at = spec.at
    length(at) == length(xs) && all(x -> any(p -> isequal(p.first, x), at), xs) ||
        throw(ArgumentError("PointPort on $(spec.var): `at` must give exactly the spatial arguments $xs."))
    perdim = map(enumerate(xs)) do (d, x)
        x0 = last(at[findfirst(p -> isequal(p.first, x), at)])
        allowed = restrict_to === nothing ? eachindex(s.grid[x]) : restrict_to[d]
        nodes, w, vol = select_port_nodes_1d(
            spec.select, s.grid[x], s.axies[x], x0, allowed, x;
            periodic = periodic === nothing ? false : periodic[d],
            upper_open = upper_open === nothing ? false : upper_open[d]
        )
        collect(zip(nodes, w, vol))
    end
    combos = vec(collect(Iterators.product(perdim...)))
    nodes = [CartesianIndex(map(c -> c[1], combo)) for combo in combos]
    weights = [prod(c -> c[2], combo) for combo in combos]
    volumes = [prod(c -> c[3], combo) for combo in combos]
    return nodes, weights, volumes
end

# Index arguments of `array` under the time derivative of `eq`: integers for an element
# equation, ranges for a slice equation, `nothing` if `eq` does not evolve `array`.
function time_derivative_index(eq, array, t)
    for d in Symbolics.get_differential_vars(eq)
        op = operation(d)
        op isa Differential && isequal(op.x, t) || continue
        v = first(arguments(d))
        iscall(v) && operation(v) === getindex && isequal(first(arguments(v)), array) ||
            continue
        return map(i -> unwrap_const(safe_unwrap(i)), arguments(v)[2:end])
    end
    return nothing
end

# Subtract `src * q_I` from the left-hand side of the equation that evolves each node.
function inject_point_source(eqs, disc, nodes, q, src, t, name, pde)
    array = first(arguments(safe_unwrap(first(disc))))
    hosted = zeros(Int, length(nodes))
    eqs = map(eqs) do eq
        idx = time_derivative_index(eq, array, t)
        idx === nothing && return eq
        inside = findall(I -> all(d -> I[d] in idx[d], eachindex(idx)), nodes)
        isempty(inside) && return eq
        hosted[inside] .+= 1
        all(i -> i isa Integer, idx) && return eq.lhs - src * q[only(inside)] ~ eq.rhs
        dims = findall(i -> !(i isa Integer), idx)
        Q = zeros(map(d -> length(idx[d]), dims)...)
        for k in inside
            Q[map(d -> findfirst(==(nodes[k][d]), idx[d]), dims)...] = q[k]
        end
        return (eq.lhs .- collect(src .* Q)) ~ eq.rhs
    end
    all(==(1), hosted) || throw(
        ArgumentError(
            "PointPort $name selects nodes $(nodes[hosted .!= 1]) that the interior equations of " *
                "$pde do not evolve exactly once. A flow into a variable whose time derivative " *
                "appears in several PDEs needs those PDEs to share the port's interior nodes."
        )
    )
    return eqs
end

# The coefficient `a` of `Dt(u)` in `pde`, zero when `pde` has no `Dt(u)`.
function port_time_coefficient(pde, u, s, name)
    a, _, linear = Symbolics.linear_expansion(pde.lhs - pde.rhs, Differential(s.time)(u))
    ok = linear && all(Symbolics.get_variables(a)) do v
        ModelingToolkitBase.isparameter(v) && !any(x -> isequal(x, v), s.x̄)
    end
    ok || throw(
        ArgumentError(
            "PointPort $name carries a flow into $u, which needs every PDE containing " *
                "Dt($u) to contain it linearly with a constant or parameter coefficient. " *
                "Got coefficient $a in $pde."
        )
    )
    return a
end

is_zero_coefficient(a) = (av = unwrap_const(safe_unwrap(a)); av isa Number && iszero(av))

function port_nodes(spec, u, s, interiormap, bcmap)
    xs = ivs(u, s)
    ifaces = map(x -> filter_interfaces(bcmap[operation(u)][x]), xs)
    periodic = map(
        (x, bs) -> any(b -> isequal(operation(b.u2), operation(u)) && isequal(b.x2, x), bs),
        xs, ifaces
    )
    upper_open = map(bs -> any(isupper, bs), ifaces)
    restrict = spec.through === nothing ? nothing : interiormap.I[interiormap.pde[u]].indices
    return select_port_nodes(spec, s; restrict_to = restrict, periodic, upper_open)
end

"""
Add the port sources and across equations that belong to `pde`, whose interior equations
are `eqs`. A port's flow is added to `Dt(u)` of its variable `u`. Several PDEs may contain
`Dt(u)`, so each gets the source scaled by its own coefficient of `Dt(u)`; the residuals
then shift by `A * e_u * src`, with `A` the matrix of time-derivative coefficients, and
the solved system gains exactly `src` in `Dt(u)`. Every PDE uses the nodes selected in the
interior of the PDE assigned to `u`. The across equation goes with that PDE.
"""
function add_point_ports(eqs, pde, interiormap, eqvar, s, derivweights, bcmap)
    across = Equation[]
    for port in derivweights.ports
        spec = port_spec(port)
        u = port_depvar(spec, s)
        own = isequal(operation(u), operation(eqvar))
        a = spec.through === nothing ? 0 : port_time_coefficient(pde, u, s, nameof(port))
        flows = !is_zero_coefficient(a)
        own || flows || continue
        nodes, w, V = port_nodes(spec, u, s, interiormap, bcmap)
        if flows
            isequal(ivs(eqvar, s), ivs(u, s)) || throw(
                ArgumentError(
                    "PointPort $(nameof(port)) carries a flow into $u, whose time derivative " *
                        "appears in $pde for $eqvar on different spatial variables."
                )
            )
            flow, scale = spec.through
            src = a * scale * getproperty(port, flow)
            eqs = inject_point_source(
                eqs, s.discvars[eqvar], nodes, w ./ V, src, s.time, nameof(port), pde
            )
        end
        own || continue
        spec.through === nothing ||
            any(p -> !is_zero_coefficient(port_time_coefficient(p, u, s, nameof(port))), keys(interiormap.I)) ||
            throw(
            ArgumentError(
                "PointPort $(nameof(port)) carries a flow into $u, but no PDE contains Dt($u)."
            )
        )
        push!(across, getproperty(port, spec.across) ~ sum(w .* [s.discvars[u][I] for I in nodes]))
    end
    return vcat(eqs, across)
end

"""
Rebuild the discretized system `sys` with its `PointPort`s as subsystems.
"""
function attach_point_ports(sys, ports; checks = true)
    scales = [port_spec(p).through[2] for p in ports if port_spec(p).through !== nothing]
    scaleps = filter(ModelingToolkitBase.isparameter, mapreduce(Symbolics.get_variables, union, scales; init = []))
    return ModelingToolkitBase.System(
        get_eqs(sys), ModelingToolkitBase.get_iv(sys), get_unknowns(sys),
        union(ModelingToolkitBase.get_ps(sys), scaleps);
        systems = ports,
        initialization_eqs = ModelingToolkitBase.get_initialization_eqs(sys),
        guesses = ModelingToolkitBase.get_guesses(sys),
        initial_conditions = ModelingToolkitBase.get_initial_conditions(sys),
        tspan = ModelingToolkitBase.get_tspan(sys),
        metadata = [
            ModelingToolkit.ProblemTypeCtx => getmetadata(sys, ModelingToolkit.ProblemTypeCtx, nothing),
        ],
        name = nameof(sys), checks
    )
end
