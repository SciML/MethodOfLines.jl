using Documenter, MethodOfLines

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"

include("pages.jl")

makedocs(sitename = "MethodOfLines.jl",
         authors = "Chris Rackauckas, Alex Jones et al.",
         clean = true, doctest = false, linkcheck = true,
         modules = [MethodOfLines],
         warnonly = true,#[:docs_block, :missing_docs, :cross_references],
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/MethodOfLines/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/MethodOfLines.jl"; push_preview = true)
