module MolecularDFT

import CrystalEnergyGrids as CEG
using Pkg: TOML

using Scratch: @get_scratch!
const MODULE_VERSION = VersionNumber(TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))["version"])
scratchspace::String = ""
function __init__()
    global scratchspace
    scratchspace = @get_scratch!("hnc-$(MODULE_VERSION.major).$(MODULE_VERSION.minor)")
end

using Unitful
using StaticArrays
using AtomsBase

include("constants.jl")
include("lammpstrj_parser.jl")
include("hypernettedchain.jl")
include("rdf.jl")
include("mdft.jl")
include("viewer.jl")
include("gridmontecarlo.jl")

end # module MolecularDFT
