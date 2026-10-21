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
