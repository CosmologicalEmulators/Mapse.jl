# The scheduling layer is ordinary Julia inside this primitive. Only serial
# chunk bodies are differentiated. No tape or cotangent graph is shared by workers.
const _ASSEMBLY_BUFFERS = (:W_buf, :W2_buf, :I1h_buf, :Pk1h_buf, :rv_eff_buf)
const _ASSEMBLY_READS = (:M, :nu_mat, :rv, :cc, :ln1pc, :Mb_vec, :fstar_vec,
                         :Pk_lin_mat, :Pk_wig_mat)
const _ASSEMBLY_PARAMS = (:eta, :k_star, :k_damp, :f_damp, :A, :sigma_v)

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(H._assemble_threaded_slices!), Matrix{Float64}, Vector{Float64},
    Vector{<:Real}, H.HMcodeWorkspace{Float64}, H.HMcodeParams{Float64},
    Float64, Bool, Union{Nothing,Real}, Matrix{Float64}, Float64, Float64,
    Float64, Bool,
}

function _threaded_array_leaves!(arrays, seen, x)
    @nospecialize x
    T = typeof(x)
    (isbitstype(T) || x isa Type || x isa Module) && return nothing
    haskey(seen, x) && return nothing
    seen[x] = nothing
    if x isa Array || (isdefined(Base, :Memory) && x isa getfield(Base, :Memory))
        isempty(x) || push!(arrays, x)
        # Storage identity is relevant even when its element payload is isbits.
        if !isbitstype(eltype(x))
            for i in eachindex(x)
                isassigned(x, i) && _threaded_array_leaves!(arrays, seen, x[i])
            end
        end
    else
        for i in 1:fieldcount(T)
            isdefined(x, i) && _threaded_array_leaves!(arrays, seen, getfield(x, i))
        end
    end
    return nothing
end

function _check_threaded_storage(p, n)
    out, k, zs, ws, pars = p[1:5]
    size(out) == (length(k), length(zs)) || throw(DimensionMismatch(
        "Threaded assembly output must have size (length(k), length(zs))"))
    reads = (k, zs, ws.M, ws.nu_mat, p[9], ws.rv, ws.cc, ws.ln1pc,
             ws.Mb_vec, ws.fstar_vec, ws.Pk_lin_mat, ws.Pk_wig_mat,
             pars.eta, pars.k_star, pars.k_damp, pars.f_damp, pars.A, pars.sigma_v)
    writes = AbstractArray[out]
    for name in _ASSEMBLY_BUFFERS, b in 1:n
        push!(writes, getfield(ws, name)[b])
    end
    for (i, a) in enumerate(writes)
        any(b -> Base.mightalias(a, b), reads) && throw(ArgumentError(
            "Threaded assembly writes must not overlap read-only inputs"))
        for j in 1:i-1
            Base.mightalias(a, writes[j]) && throw(ArgumentError(
                "Threaded assembly output and scratch buffers must be disjoint"))
        end
        for name in _ASSEMBLY_BUFFERS, b in n+1:length(getfield(ws, name))
            Base.mightalias(a, getfield(ws, name)[b]) && throw(ArgumentError(
                "Active and inactive threaded scratch buffers must be disjoint"))
        end
    end
    # Mooncake deduplicates Array objects, not their backing storage. Preserve
    # the full caller-graph contract even for fields omitted by worker contexts:
    # keep repeated-object aliases, but reject distinct overlapping headers.
    arrays = AbstractArray[]
    _threaded_array_leaves!(arrays, IdDict{Any,Nothing}(), p)
    for (i, a) in enumerate(arrays), j in 1:i-1
        Base.mightalias(a, arrays[j]) && throw(ArgumentError(
            "Threaded reverse assembly does not support distinct array headers with overlapping storage"))
    end
    return nothing
end

function _chunk_coduals(p)
    # One construction for the tuple preserves aliases across its arguments.
    d = Mooncake.tangent(Mooncake.zero_fcodual(p))
    return map(Mooncake.CoDual, p, d)
end

@inline _assembly_fields(getter, x, names) =
    NamedTuple{names}(map(name -> getter(x, name), names))

function _compact_chunk_arguments(p, chunk)
    shared = _assembly_fields(getfield, p[4], _ASSEMBLY_READS)
    params = _assembly_fields(getfield, p[5], _ASSEMBLY_PARAMS)
    buffers = NamedTuple{_ASSEMBLY_BUFFERS}(
        map(name -> getfield(p[4], name)[chunk], _ASSEMBLY_BUFFERS))
    return (p[1:3]..., shared, params, p[6:13]..., buffers)
end

function _compact_chunk_fdata(args, chunk)
    fd = map(Mooncake.tangent, args)
    shared = _assembly_fields(Mooncake._get_fdata_field, fd[4], _ASSEMBLY_READS)
    params = _assembly_fields(Mooncake._get_fdata_field, fd[5], _ASSEMBLY_PARAMS)
    buffers = NamedTuple{_ASSEMBLY_BUFFERS}(
        map(name -> Mooncake._get_fdata_field(fd[4], name)[chunk], _ASSEMBLY_BUFFERS))
    return (fd[1:3]..., shared, params, fd[6:13]..., buffers)
end

function _chunk_forward(rule, args, chunk, nchunks)
    _, back = rule(Mooncake.zero_fcodual(H._assemble_owned_slice_chunk!), args...,
                   Mooncake.zero_fcodual(chunk), Mooncake.zero_fcodual(nchunks))
    return back
end

_chunk_reverse(back) = back(Mooncake.NoRData())

function _seed_chunk!(private, original_fd, chunk, nchunks, nz)
    source = original_fd[1]
    _, dest = Mooncake.arrayify(private[1])
    columns = fld((chunk-1)*nz, nchunks)+1:fld(chunk*nz, nchunks)
    copyto!(view(dest, :, columns), view(source, :, columns))
    private_buffers = Mooncake.tangent(private[14])
    for name in _ASSEMBLY_BUFFERS
        copyto!(getfield(private_buffers, name), getfield(original_fd[14], name))
    end
    return nothing
end

function _clear_assembly_outputs!(args, n)
    _, doutput = Mooncake.arrayify(args[1])
    fill!(doutput, 0.0)
    ws = Mooncake.primal(args[4])
    for name in _ASSEMBLY_BUFFERS, chunk in 1:n
        buffers = Mooncake.get_tangent_field(Mooncake.tangent(args[4]), name)
        _, d = Mooncake.arrayify(getfield(ws, name)[chunk], buffers[chunk])
        fill!(d, 0.0)
    end
    return nothing
end

function Mooncake.rrule!!(::Mooncake.CoDual{typeof(H._assemble_threaded_slices!)},
                         args::Vararg{Mooncake.CoDual,13})
    p = map(Mooncake.primal, args)
    n = H._assembly_chunk_count(p[3], p[4])
    if n == 0
        return Mooncake.zero_fcodual(nothing), _ ->
            (Mooncake.NoRData(), map(Mooncake.zero_rdata, p)...)
    end
    _check_threaded_storage(p, n)
    # build_rrule returns independent copied rule state on cache hits. Build the
    # top-level rules outside Polyester; lazy descendants may compile on first use.
    inputs = [_compact_chunk_arguments(p, i) for i in 1:n]
    rules = [Mooncake.build_rrule(H._assemble_owned_slice_chunk!, inputs[i]..., i, n) for i in 1:n]
    private = [_chunk_coduals(input) for input in inputs]
    # Only the task-level dispatch is erased; the numerical chunk is specialized.
    backs = Vector{Any}(undef, n)
    @batch for i in 1:n
        backs[i] = _chunk_forward(rules[i], private[i], i, n)
    end
    function threaded_assembly_pullback(_)
        # Project current caller cotangents at pullback entry, not forward time.
        # These are references into the full graph, including omitted-field aliases.
        original_fd = [_compact_chunk_fdata(args, i) for i in 1:n]
        for i in 1:n
            _seed_chunk!(private[i], original_fd[i], i, n, length(p[3]))
        end
        _clear_assembly_outputs!(args, n)
        results = Vector{Any}(undef, n)
        @batch for i in 1:n
            results[i] = _chunk_reverse(backs[i])
        end
        rdata = map(Mooncake.zero_rdata, p)
        for i in 1:n
            # One alias-aware tuple reduction, not one accumulation per argument.
            Mooncake.increment!!(original_fd[i], map(Mooncake.tangent, private[i]))
            rdata = Mooncake.increment!!(rdata, results[i][2:14])
        end
        return (Mooncake.NoRData(), rdata...)
    end
    return Mooncake.zero_fcodual(nothing), threaded_assembly_pullback
end
