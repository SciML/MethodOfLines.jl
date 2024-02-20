# Performance Tips

While the default ModelingTolkit backend is highly capable, it doesn't scale well to very large systems of ODEs, present iun high resolution and/or higher dimensional PDE semidiscretizations that MethodOfLines.jl is capable of generating.

To overcome this limitation, and scale MOL to realistic physics simulations, JuliaHub Inc. has released [JuliaSimCompiler.jl (installation instructions here)](https://help.juliahub.com/juliasimcompiler/dev/).

Whenever this package and MethodOfLines are loaded together, MOL will also load `MethodOfLinesJuliaSimCompilerExt.jl`, an extension that takes advantage of this backend to speed up your compilation and solves automatically, with the same interface you are used to.

!!! note
    
    JuliaSimCompiler is part of JuliaSim and thus requires a valid JuliaSim license to use. JuliaSim is a proprietary software developed by JuliaHub Inc. Using the packages through the registry requires a valid JuliaSim license. It is free to use for non-commercial academic teaching and research purposes. For commercial users, license fees apply. Please refer to the [End User License Agreement](https://juliahub.com/company/eula/?_gl=1*120lqg6*_ga*MTAxODQ4OTE3Mi4xNjk0MDA4MDM5*_ga_8FC7JQQLXX*MTY5NDAwODAzOC4xLjEuMTY5NDAwODgxMC4wLjAuMA..) for details. Please contact [sales@juliahub.com](https://juliahub.com/products/pricing/) for purchasing information.
