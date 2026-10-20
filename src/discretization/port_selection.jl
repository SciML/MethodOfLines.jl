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

function select_port_nodes_1d(
        sel, grid, domain, x0, allowed, x; periodic = false, upper_open = false
    )
    tol = sqrt(eps(Float64)) * (last(domain) - first(domain))
    first(domain) - tol <= x0 <= last(domain) + tol ||
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
        first(xs) - tol <= x0 <= last(xs) + tol || throw(
            ArgumentError(
                "PointPort coordinate $x = $x0 carries a flow but lies outside the interior nodes " *
                    "$x ∈ [$(first(xs)), $(last(xs))] that hold the PDE; the nodes nearer the boundary " *
                    "hold boundary conditions. Put the flow in a flux boundary condition instead, " *
                    "e.g. `-k * Dx(T(t, 0)) ~ heater.Q_flow / A`."
            )
        )
    end
    n <= length(unique(idx)) ||
        throw(ArgumentError("PointPort selection $sel needs $n nodes in $x but only $(length(unique(idx))) are available."))
    hit = findfirst(xi -> abs(xi - x0) <= tol, xs)
    lo = clamp(searchsortedlast(xs, x0), 1, length(xs) - 1)
    window = if hit !== nothing
        hit:hit
    elseif n == 1
        inside = k -> first(domain) <= xs[k] <= last(domain)
        dlo, dhi = x0 - xs[lo], xs[lo + 1] - x0
        pick = abs(dlo - dhi) <= tol ? (inside(lo) || !inside(lo + 1) ? lo : lo + 1) :
            (dlo < dhi ? lo : lo + 1)
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
function inject_point_source(eqs, disc, nodes, q, src, t, name)
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
    all(==(1), hosted) || error(
        "Internal error: PointPort $name selected nodes $(nodes[hosted .!= 1]) that are not " *
            "evolved by exactly one interior equation."
    )
    return eqs
end

# The coefficient `a` of `Dt(eqvar)` in `pde`, so that `a * src` adds `src` to `Dt(eqvar)`.
function port_time_coefficient(pde, eqvar, s, name)
    a, _, linear = Symbolics.linear_expansion(pde.lhs - pde.rhs, Differential(s.time)(eqvar))
    av = unwrap_const(safe_unwrap(a))
    ok = linear && !(av isa Number && iszero(av)) &&
        all(Symbolics.get_variables(a)) do v
        ModelingToolkitBase.isparameter(v) && !any(x -> isequal(x, v), s.x̄)
    end
    ok || throw(
        ArgumentError(
            "PointPort $name carries a flow into $eqvar, which needs a first-order time " *
                "derivative of $eqvar entering its PDE linearly with a constant or " *
                "parameter coefficient. Got coefficient $a in $pde."
        )
    )
    return a
end

"""
Add the sources and across equations of the ports on `eqvar` to its interior equations.
"""
function add_point_ports(eqs, pde, interior, eqvar, s, derivweights, bcmap)
    ports = filter(p -> isequal(operation(port_spec(p).var), operation(eqvar)), derivweights.ports)
    isempty(ports) && return eqs
    disc = s.discvars[eqvar]
    ifaces = map(x -> filter_interfaces(bcmap[operation(eqvar)][x]), ivs(eqvar, s))
    periodic = map(
        (x, bs) -> any(b -> isequal(operation(b.u2), operation(eqvar)) && isequal(b.x2, x), bs),
        ivs(eqvar, s), ifaces
    )
    upper_open = map(bs -> any(isupper, bs), ifaces)
    across = map(ports) do port
        spec = port_spec(port)
        restrict = spec.through === nothing ? nothing : interior.indices
        nodes, w, V = select_port_nodes(
            spec, s; restrict_to = restrict, periodic, upper_open
        )
        if spec.through !== nothing
            flow, scale = spec.through
            a = port_time_coefficient(pde, eqvar, s, nameof(port))
            src = a * scale * getproperty(port, flow)
            eqs = inject_point_source(eqs, disc, nodes, w ./ V, src, s.time, nameof(port))
        end
        getproperty(port, spec.across) ~ sum(w .* [disc[I] for I in nodes])
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
