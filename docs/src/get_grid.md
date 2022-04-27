# get_discrete (@id get_grid)

`MethodOfLines.jl` exports a helper function `get_discrete`, which returns a `Dict` with the keys being the independent and dependent variables, and the values their corresponding discrete grid, and discretized variables used in the discretization. It is used as following:
```julia
grid = get_discrete(pdesys, discretization)
discrete_x = grid[x]
# Retrieve shaped solution
u_sol = [map(d -> sol[d][i], grid[u(t, x)]) for i in 1:length(sol[t])]
```