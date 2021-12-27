struct DifferentialDiscretizer{T}
    map::Dict{Num, DerivativeOperator}
    orders::Vector{T}
end

function DifferentialDiscretizer(pde, s, discretization)
    approx_order = discretization.approx_order
    d_orders(x) = reverse(sort(collect(union(differential_order(pde.rhs, x), differential_order(pde.lhs, x)))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]
    differentialmap = Array{Pair{Num,DerivativeOperator},1}()
    orders = Int[]
    # Hardcoded to centered difference, generate weights for each differential
    # TODO: Add handling for upwinding
    for (j,x) in enumerate(s.nottime)
        push!(orders, d_orders(x))
        rs = [(Differential(x)^d) => DifferentialDiscretizer{j}(d, approx_order, s.dxs[j],length(s.grid[j])) for d in orders[j]]
        for r in rs
            push!(differentialmap, r)
        end
    end
    return DifferentialDiscretizer{eltype(orders)}(Dict(differentialmap), orders)
end