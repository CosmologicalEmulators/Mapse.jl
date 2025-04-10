using Documenter
using Mapse

ENV["GKSwstype"] = "100"

push!(LOAD_PATH,"../src/")

makedocs(
    modules = [Mapse],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
    sidebar_sitename=true),
    sitename = "Mapse.jl",
    authors  = "Marco Bonici",
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/CosmologicalEmulators/Mapse.jl.git",
    devbranch = "develop"
)
