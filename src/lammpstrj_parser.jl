using Base.Threads

struct Frame <: AbstractVector{SVector{3,Float64}}
    atoms::Vector{Tuple{Int,SVector{3,Float64}}}
    unitcell::SMatrix{3,3,Float64,9}
end
Base.size(f::Frame) = (length(f.atoms),)
Base.getindex(f::Frame, i::Int) = f.atoms[i][2]

struct LAMMPStrj <: AbstractVector{Frame}
    file::String
    timesteps::Vector{Int}
end
function LAMMPStrj(file)
    io = open(file)
    timesteps = Int[]
    seekend(io)
    n = position(io)
    seekstart(io)
    l0 = readline(io)
    @assert l0 == "ITEM: TIMESTEP"
    pstart = position(io)
    push!(timesteps, pstart)
    p = pstart
    while p < n
        l = readline(io)
        p = position(io)
        if l == "ITEM: TIMESTEP"
            break
        end
    end
    k = 1
    @assert p < n
    push!(timesteps, p)
    hint = div((p - pstart - 16)*995, 1000)
    seek(io, position(io) + hint)
    while p < n
        l = readline(io)
        p = position(io)
        if l == "ITEM: TIMESTEP"
            if p - last(timesteps) > 7/5*hint # suspiciously large step
                p = last(timesteps)+1
                print('.')
            else
                push!(timesteps, p)
                k += 1
                hint = div((p - pstart - 16*k)*995, 1000*k)
                p += hint
            end
            seek(io, p)
        end
    end
    LAMMPStrj(file, timesteps)
end

Base.size(trj::LAMMPStrj) = (length(trj.timesteps),)
function Base.getindex(trj::LAMMPStrj, step::Int)
    io = open(trj.file)
    seek(io, trj.timesteps[step])
    readline(io); readline(io);
    n = parse(Int, readline(io))
    atoms = Vector{Tuple{Int,SVector{3,Float64}}}(undef, n)
    readline(io)
    x1, x2 = parse.(Float64, split(readline(io)))
    y1, y2 = parse.(Float64, split(readline(io)))
    z1, z2 = parse.(Float64, split(readline(io)))
    format = split(readline(io))
    unitcell = SMatrix{3,3,Float64,9}([x2-x1  0.0    0.0
                        0.0    y2-y1  0.0
                        0.0    0.0    z2-z1])
    @assert format[2] == "ATOMS"
    id = type = x = y = z = 0
    for (i, fmt) in enumerate(format)
        if fmt == "id"
            id = i-2
        elseif fmt == "type"
            type = i-2
        elseif fmt == "x"
            x = i-2
        elseif fmt == "y"
            y = i-2
        elseif fmt == "z"
            z = i-2
        end
    end
    for _ in 1:n
        l = split(readline(io))
        i = parse(Int, l[id])
        atoms[i] = (parse(Int, l[type]), SVector{3,Float64}(parse(Float64, l[x]), parse(Float64, l[y]), parse(Float64, l[z])))
    end
    Frame(atoms, unitcell)
end
