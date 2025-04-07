using Documenter, MethodOfLines

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"

include("pages.jl")

makedocs(sitename = "MethodOfLines.jl",
         authors = "Chris Rackauckas, Alex Jones et al.",
         clean = true,
         doctest = false,
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         modules = [MethodOfLines],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/MethodOfLines/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/MethodOfLines.jl"; push_preview = true)
