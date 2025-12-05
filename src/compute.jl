using StaticArrays


function henon_rule(u, p, n) # here `n` is "time", but we don't use it.
    x, y = u # system state
    a, b = p # system parameters
    xn = a - x^2 + b*y
    yn = x
    return SVector(xn, yn)
end


function _get_basin_entropy(a, b, grid, grid_rec; consecutive_recurrences = 50000, show_progress = false) 
        ds = DeterministicIteratedMap(henon_rule, rand(2), [a,b])
        mapper = AttractorsViaRecurrences(ds, grid_rec;  consecutive_recurrences)
        bas, _ = basins_of_attraction(mapper, grid; show_progress) 
        Sb,Sbb = basin_entropy(Int64.(bas), 20) 
        return Sb, Sbb, bas
end


function henon_entropy(d)
    @unpack  b, ai, af, al, res = d
    a = range(ai, af, length = al)
    Sb = zeros(length(a))
    Sbb = zeros(length(a))
    xg = yg = range(-2,2,length = res)
    grid = (xg,yg) 
    xg = yg = range(-4,4,length = 10001)
    grid_rec = (xg,yg) 

    @Threads.threads for k in eachindex(a)
        @show a[k]
        Sb[k], Sbb[k], _ = _get_basin_entropy(a[k],b, grid, grid_rec)
    end
    return @strdict(Sb, Sbb, a)
end


function henon_continuation(d)
    @unpack  b, ai, af, al = d
    a = range(ai, af, length = al)
    ds = DeterministicIteratedMap(henon_rule, rand(2), [a[1],b])
    xg = yg = range(-3,3,length = 4001)
    mapper = AttractorsViaRecurrences(ds, (xg, yg); Ttr=5000, consecutive_recurrences = 10000)
    pidx = 1; spp = 50
    sampler, = statespace_sampler(HRectangle([-5, -5], [5, 5]))
    ## RECURENCE CONTINUATION
    cnt = RecurrencesFindAndMatch(mapper)
    fs, att = global_continuation(
            cnt, a, pidx, sampler;
            show_progress = true, 
            samples_per_parameter = spp
            )
    return @strdict(fs, att)
end


function get_branches(b, ai, af, al,att)
    a = range(ai, af, length = al)
    s = Vector{Int32}[]
    for e in att
        push!(s, collect(keys(e)))
    end
    s = unique(vcat(s...))
    branches = Dict( s[k] => Vector{Vector{Float64}}() for k in 1:length(s))

    # ptlst = Vector{Vector{Float64}}()
    ds = DeterministicIteratedMap(henon_rule, rand(2), [a[1], b])
    T = 2000
    Ttr = 500;
    for (k,el) in enumerate(att)
            for p in el
                set_parameter!(ds, 1, a[k])
                tra,t = trajectory(ds, T, p[2][1]; Ttr)
                  for y in tra[1900:2000]
                     v = [a[k], y[1], y[2]]
                     push!(branches[p[1]], v)
                  end
            end
    end
    branches = Dict(k => StateSpaceSet(branches[k]) for k in keys(branches))
    return branches
end


function get_bif_points(branches::Dict)
    s = collect(keys(branches))
    bif_points = StateSpaceSet(branches[s[1]])
    for k in 2:length(s)
        append!(bif_points, StateSpaceSet(branches[s[k]]))
    end
    return bif_points
end



function get_bif(b, ai, af, al; force = false)
    d = @strdict  b ai af al
    data, file = produce_or_load(
        datadir("basins"), 
        d, 
        henon_continuation;
        prefix = "henon_atr", storepatch = false,
        suffix = "jld2", force = force
    )
    return data
end

function get_entropy(b, ai, af, al, res; force = false)
    d = @strdict  b ai af al res
    data, file = produce_or_load(
        datadir("basins"), 
        d, 
        henon_entropy;
        prefix = "henon_entropy", storepatch = false,
        suffix = "jld2", force = force
    )
    return data
end


# Compute initial basin, this will be seed for the priors: 

function get_fresh_basins(a, b, grid, grid_rec, atts = nothing; consecutive_recurrences = 50000, show_progress = false) 
    rmap = nothing
    mapper = get_mapper(a, b, grid_rec, atts; consecutive_recurrences)
    bas, _ = basins_of_attraction(mapper, grid; show_progress) 

    return bas, mapper, rmap 
end

function seed_mapper!(mapper, prev_attractors)
    for att in values(prev_attractors)
        for u0 in seeding(att)
            label = mapper(u0; show_progress = false)
        end
    end
end


function get_mapper(a, b, grid_rec, atts = nothing; consecutive_recurrences = 50000)
    ds = DeterministicIteratedMap(henon_rule, rand(2), [a,b])
    mapper = AttractorsViaRecurrences(ds, grid_rec;  consecutive_recurrences)
    if !isnothing(atts) && !isempty(atts)
        seed_mapper!(mapper, atts)
        mtch = MatchBySSSetDistance(; distance = Hausdorff(), threshold = Inf, use_vanished = false)
        rmap = matching_map!(mapper.bsn_nfo.BoA.attractors, atts, mtch)
    end
    return mapper
end

function seeding(attractor::AbstractStateSpaceSet)
    return (attractor[1],) # must be iterable
end
