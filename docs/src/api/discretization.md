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
Filter = t -> !(t in (PointPort, Nearest, Interpolate))
```

The connector port types are documented on the [connector ports API page](connectors.md).

## Developer API

The private API reference is available on the [developer API page](developer.md).
