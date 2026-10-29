```@meta
CurrentModule = MethodOfLines
```

# Discretization

## Public API

```@autodocs
Modules = [MethodOfLines]
Public = true
Private = false
Order = [:constant, :type, :function]
Pages = [
    "MOL_discretization.jl", "MOL_utils.jl", "discretization/schemes/WENO/WENO.jl",
    "interface/MOLFiniteDifference.jl", "interface/PseudospectralDiscretization.jl",
    "interface/callbacks.jl", "interface/grid_types.jl", "interface/scheme_types.jl",
]
```

The connector port types are documented on the [connector ports API page](connectors.md).

## Developer API

The private API reference is available on the [developer API page](developer.md).
