struct BoundaryMap
    vars
    derivatives
    initial
end

struct DerivativeMap{N}
    map
    orders
end

count(::DerivativeMap{N}) where N = N

function generate_maps(bcs, t, s, tspan, grid_align)
    if grid_align == center_align
        # depvarbcmaps will dictate what to replace the variable terms with in the bcs
        # replace u(t,0) with u₁, etc
        depvarbcmaps = reduce(vcat,[subvar(depvar, edgevals(s)) .=> edgevar for (depvar, edgevar) in zip(s.vars, edgevars(s))])
    end
    # depvarderivbcmaps will dictate what to replace the Differential terms with in the bcs
    if length(s.nottime) == 1
        # 1D system
        bc_der_orders = filter(!iszero,sort(unique([count_differentials(bc.lhs, s.nottime[1]) for bc in bcs])))
        left_weights(d_order,j) = calculate_weights(d_order, s.grid[j][1], s.grid[j][1:1+d_order])
        right_weights(d_order,j) = calculate_weights(d_order, s.grid[j][end], s.grid[j][end-d_order:end])
        # central_neighbor_idxs(II,j) = [II-CartesianIndex((1:length(s.nottime).==j)...),II,II+CartesianIndex((1:length(s.nottime).==j)...)]
        left_idxs(d_order) = CartesianIndices(s.grid[1])[1:1+d_order]
        right_idxs(d_order,j) = CartesianIndices(s.grid[j])[end-d_order:end]
        # Constructs symbolic spatially discretized terms of the form e.g. au₂ - bu₁
        derivars = [[[dot(left_weights(d,1), depvar[left_idxs(d)]), dot(right_weights(d,1), depvar[right_idxs(d,1)])] for d in bc_der_orders] for depvar in s.discvars]
        # Create list of all the symbolic Differential terms evaluated at boundary e.g. Differential(x)(u(t,0))
        subderivar(depvar,d_order,x) = substitute.(((Differential(x)^d_order)(depvar),), edgevals(s))
        # Create map of symbolic Differential terms with symbolic spatially discretized terms
        depvarderivbcmaps = []
        for x in s.nottime
            rs = (subderivar(depvar,bc_der_orders[j],x) .=> derivars[i][j] for j in 1:length(bc_der_orders), (i,depvar) in enumerate(s.vars))
            for r in rs
                push!(depvarderivbcmaps, r)
            end
        end

        if grid_align == edge_align
            # Constructs symbolic spatially discretized terms of the form e.g. (u₁ + u₂) / 2
            bcvars = [[dot(ones(2)/2,depvar[left_idxs(1)]), dot(ones(2)/2,depvar[right_idxs(1,1)])]
                    for depvar in s.discvars]
            # replace u(t,0) with (u₁ + u₂) / 2, etc
            depvarbcmaps = reduce(vcat,[subvar(depvar, edgevals(s)) .=> bcvars[i]
                                for (i, depvar) in enumerate(s.vars) for param in s.nottime])

        end
    else
        # Higher dimension
        # TODO: Fix Neumann and Robin on higher dimension
        #! Use Rules
        depvarderivbcmaps = []
    end

    if t === nothing
        initmaps = s.vars
    else
        initmaps = substitute.(s.vars,[t=>tspan[1]])
    end

    # All unique order of derivates in BCs
    bc_der_orders = filter(!iszero,sort(unique([count_differentials(bc.lhs, s.nottime[1]) for bc in bcs])))
    # no. of different orders in BCs
    n = length(bc_der_orders)
    return BoundaryMap(depvarbcmaps, DerivativeMap{n}(depvarderivbcmaps, bc_der_orders), initmaps)
end
 
### INITIAL AND BOUNDARY CONDITIONS ###
"""
Mutates bceqs and u0 by finding relevant equations and discretizing them
"""
function generate_u0_and_bceqs!!(u0, bceqs, bcs, s, depvar_ops, tspan, discretization)
    grid_align = discretization.grid_align
    t=discretization.time
    boundarymap = generate_maps(bcs, t, s, tspan, grid_align)
      # Generate initial conditions and bc equations
    for bc in bcs
        bcdepvar = first(get_depvars(bc.lhs, depvar_ops))
        if any(u->isequal(operation(u),operation(bcdepvar)), s.vars)
            if t !== nothing && operation(bc.lhs) isa Sym && !any(x -> isequal(x, t.val), arguments(bc.lhs))
                # initial condition
                # Assume in the form `u(...) ~ ...` for now
                initindex = findfirst(isequal(bc.lhs), boundarymap.initial) 
                if initindex !== nothing
                    push!(u0,vec(s.discvars[initindex] .=> substitute.((bc.rhs,),gridvals(s))))
                end
            else
                # Algebraics.vars equations for BCs
                initindex = findfirst(x->occursin(x, bc.lhs), first.(boundarymap.vars))
                if initindex !== nothing
                    bcargs = arguments(first(boundarymap.vars[initindex]))
                    # Replace Differential terms in the bc lhs with the symbolic spatially discretized terms
                    # TODO: Fix Neumann and Robin on higher dimension
                    # Update: Fixed for 1D systems
                    derivativecount = count_differentials(bc.lhs, s.nottime[1])
                    derivativeindex = findfirst(isequal(derivativecount), boundarymap.derivatives.orders)
                    k = initindex%2 == 0 ? 2 : 1

                    if (length(s.nottime) == 1) && (derivativeindex !== nothing)
                        lhs = substitute(bc.lhs, boundarymap.derivatives.map[derivativeindex + count(boundarymap.derivatives)*Int(floor((initindex-1)/2))][k]) 
                    else
                        lhs = bc.lhs
                    end
                        
                    # Replace symbol in the bc lhs with the spatial discretized term
                    lhs = substitute(lhs,boundarymap.vars[initindex])
                    rhs = substitute.((bc.rhs,), edgemaps(s)[bcargs])
                    lhs = lhs isa Vector ? lhs : [lhs] # handle 1D
                    push!(bceqs,lhs .~ rhs)
                end
            end
        end
    end
end

#function generate_u0_and_bceqs_with_rules!!(u0, bceqs, bcs, t, s, depvar_ops)