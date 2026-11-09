using Test, Mapse, Mooncake, ForwardDiff, DifferentiationInterface

function _threaded_boundary_fixture(; feedback=false, alias_inputs=false, alias_omitted=false,
                                    use_fast_specials=true)
    H = Mapse.HMcode
    k = collect(10.0 .^ range(-3, 1; length=8))
    z = collect(range(0.0, 1.0; length=max(5, Threads.nthreads()+1)))
    mass = collect(10.0 .^ range(8, 15; length=8))
    cosmo = Mapse.HMCodeCosmology(0.31, 0.049, 0.67, 0.965, 0.81,
                                -0.93, 0.12, 0.0014, 0.01)
    sigma = (r, z) -> 5.0 / (1.0+r) / (1.0+z)
    pk = (q, z) -> 1e4*(q/0.1)^0.96/(1+(q/0.1)^2)^2/(1+z)^2
    ws = H.HMcodeWorkspace(k, z, mass, cosmo, sigma, pk;
                           nthreads=Threads.maxthreadid()+1)
    out = ws.Ptmp1 # Intentional output/workspace alias, as in feedback assembly.
    temp = feedback ? 10.0^7.8 : nothing
    pars = feedback ? ws.params_notweaks : ws.params_tweaks
    H.hmcode_power_single!(out, k, z, cosmo, ws; T_AGN=temp,
                           tweaks=!feedback, threaded=false, hmpars=pars)
    for name in (:W_buf, :W2_buf, :I1h_buf, :Pk1h_buf, :rv_eff_buf)
        for (i, b) in enumerate(getfield(ws, name))
            fill!(b, 0.2+i)
        end
    end
    fill!(out, 7.3)
    weights = feedback ? ws.w1h_mat_fb : ws.w1h_mat
    if alias_inputs
        # Exercise a shared read-only leaf appearing both as a direct argument
        # and a workspace field. Only the assembly boundary is under test here.
        ws.M .= k
        k = ws.M
    end
    if alias_omitted
        # Selected read/write leaves also occur in fields omitted by the private
        # context. Their original cotangents must remain linked by identity.
        ws.R = ws.M
        for i in eachindex(ws.rv_eff_buf)
            ws.gcol_buf[i] = ws.rv_eff_buf[i]
        end
    end
    return (out, k, z, ws, pars, H._hmcode_matter_density(cosmo.Omega_m),
            !feedback, temp, weights, cosmo.Omega_m,
            cosmo.Omega_m-cosmo.Omega_b-cosmo.Omega_nu, cosmo.Omega_b, use_fast_specials)
end

function _serial_owned_chunks!(n, args...)
    H = Mapse.HMcode
    for chunk in 1:n
        H._assemble_slice_chunk!(args..., chunk, n)
    end
    return nothing
end

function _threaded_test_coduals(p)
    return map(Mooncake.CoDual, p, Mooncake.tangent(Mooncake.zero_fcodual(p)))
end

function _threaded_test_arrays(p)
    ws, pars = p[4], p[5]
    arrays = AbstractArray[p[1], p[2], p[3], p[9]]
    for name in fieldnames(typeof(ws))
        a = getfield(ws, name)
        # Interpolations also subtype AbstractArray, but their tangent is a
        # structured object rather than a writable array of numeric cotangents.
        a isa Array{<:Number} && push!(arrays, a)
    end
    append!(arrays, [getfield(pars, name) for name in fieldnames(typeof(pars))])
    for params in (ws.params_tweaks, ws.params_notweaks)
        append!(arrays, [getfield(params, name) for name in fieldnames(typeof(params))])
    end
    for name in (:W_buf, :W2_buf, :I1h_buf, :Pk1h_buf, :rv_eff_buf, :gcol_buf, :wtcol_buf)
        append!(arrays, getfield(ws, name))
    end
    return arrays
end

function _threaded_test_bars(args)
    wsbar, parbar = Mooncake.tangent(args[4]), Mooncake.tangent(args[5])
    ws = Mooncake.primal(args[4])
    arrays = AbstractArray[Mooncake.arrayify(args[i])[2] for i in (1,2,3,9)]
    for name in fieldnames(typeof(ws))
        if getfield(ws, name) isa Array{<:Number}
            push!(arrays, Mooncake.get_tangent_field(wsbar, name))
        end
    end
    for name in fieldnames(typeof(Mooncake.primal(args[5])))
        push!(arrays, Mooncake._get_fdata_field(parbar, name))
    end
    for name in (:params_tweaks, :params_notweaks)
        paramsbar = Mooncake._get_fdata_field(wsbar, name)
        for field in fieldnames(typeof(getfield(ws, name)))
            push!(arrays, Mooncake._get_fdata_field(paramsbar, field))
        end
    end
    for name in (:W_buf, :W2_buf, :I1h_buf, :Pk1h_buf, :rv_eff_buf, :gcol_buf, :wtcol_buf)
        append!(arrays, Mooncake.get_tangent_field(wsbar, name))
    end
    return arrays
end

function _independent_threaded_boundary(p)
    before = copy.(_threaded_test_arrays(p))
    args = _threaded_test_coduals(p)
    _, back = Mooncake.rrule!!(
        Mooncake.zero_fcodual(Mapse.HMcode._assemble_threaded_slices!), args...)
    prediction = copy(p[1])
    fill!(Mooncake.arrayify(args[1])[2], 1.0)
    back(Mooncake.NoRData())
    restored = all(a == b for (a,b) in zip(_threaded_test_arrays(p), before))
    return prediction, copy(Mooncake.arrayify(args[2])[2]), restored
end

function _repeated_threaded_boundary(p, repeats)
    world = Base.get_world_counter()
    first = _independent_threaded_boundary(p)
    for iteration in 2:repeats
        Base.get_world_counter() == world || error("World advanced during concurrent validation")
        _independent_threaded_boundary(p) == first || error("Concurrent replay changed at $iteration")
    end
    return first
end

function _concurrent_threaded_boundaries(inputs; repeats=1)
    # Define all task callbacks before execution, not between top-level spawns.
    world = Base.get_world_counter()
    tasks = Task[]
    try
        @sync for p in inputs
            push!(tasks, Threads.@spawn _repeated_threaded_boundary(p, repeats))
        end
        return map(fetch, tasks)
    finally
        println("CONCURRENT_WORLD before=$world after=$(Base.get_world_counter()) repeats=$repeats tasks=$(length(inputs))")
    end
end

# In the fresh hmcode-threaded focused process these are the first chunk-rule
# calls. The reference is deliberately evaluated only AFTER both callers finish.
@testset "Cold concurrent independent threaded reverse states" begin
    p = _threaded_boundary_fixture(; feedback=true)
    p1, p2 = deepcopy(p), deepcopy(p)
    first, second = _concurrent_threaded_boundaries((p1,p2))
    expected = _independent_threaded_boundary(p)
    @test first == expected
    @test second == expected
    @test expected[3]
end

@testset "Compact worker context structure and alias projections" begin
    ext = Base.get_extension(Mapse, :MapseMooncakeExt)
    p = _threaded_boundary_fixture(; feedback=true, alias_inputs=true, alias_omitted=true)
    original = _threaded_test_coduals(p)
    n = Mapse.HMcode._assembly_chunk_count(p[3], p[4])
    contexts = [ext._compact_chunk_arguments(p, i) for i in 1:n]
    private = [ext._chunk_coduals(context) for context in contexts]
    for i in 1:n
        context = contexts[i]
        projected = ext._compact_chunk_fdata(original, i)
        @test length(context) == 14
        @test keys(context[4]) == (:M, :nu_mat, :rv, :cc, :ln1pc, :Mb_vec,
                                   :fstar_vec, :Pk_lin_mat, :Pk_wig_mat)
        @test keys(context[5]) == (:eta, :k_star, :k_damp, :f_damp, :A, :sigma_v)
        @test keys(context[14]) == (:W_buf, :W2_buf, :I1h_buf, :Pk1h_buf, :rv_eff_buf)
        @test typeof(projected) == typeof(map(Mooncake.tangent, private[i]))
        for cd in private[i]
            @test isnothing(Mooncake.verify_fdata_type(typeof(Mooncake.primal(cd)),
                                                       typeof(Mooncake.tangent(cd))))
        end
        @test context[2] === context[4].M
        @test Mooncake.tangent(private[i][2]) === Mooncake.tangent(private[i][4]).M
        @test projected[2] === projected[4].M === Mooncake.get_tangent_field(
            Mooncake.tangent(original[4]), :R)
        for name in keys(context[14])
            @test getfield(context[14], name) === getfield(p[4], name)[i]
            @test getfield(projected[14], name) === Mooncake.get_tangent_field(
                Mooncake.tangent(original[4]), name)[i]
            for j in 1:i-1
                @test getfield(Mooncake.tangent(private[i][14]), name) !==
                      getfield(Mooncake.tangent(private[j][14]), name)
            end
        end
        @test projected[14].rv_eff_buf === Mooncake.get_tangent_field(
            Mooncake.tangent(original[4]), :gcol_buf)[i]
    end
end

@testset "Threaded assembly ownership and full mutation contract" begin
    for feedback in (false, true), alias_inputs in (false, true),
            alias_omitted in (false, true), use_fast_specials in (true, false)
        p = _threaded_boundary_fixture(; feedback, alias_inputs, alias_omitted, use_fast_specials)
        q = deepcopy(p)
        before = copy.(_threaded_test_arrays(p))
        serial_args, parallel_args = _threaded_test_coduals(q), _threaded_test_coduals(p)
        n = Mapse.HMcode._assembly_chunk_count(q[3], q[4])
        serial_rule = Mooncake.build_rrule(_serial_owned_chunks!, n, q...)
        _, serial_back = serial_rule(Mooncake.zero_fcodual(_serial_owned_chunks!),
                                    Mooncake.zero_fcodual(n), serial_args...)
        _, parallel_back = Mooncake.rrule!!(
            Mooncake.zero_fcodual(Mapse.HMcode._assemble_threaded_slices!), parallel_args...)
        for (a, b) in zip(_threaded_test_arrays(p), _threaded_test_arrays(q))
            @test a ≈ b rtol=8eps(Float64) atol=1e-12
        end
        # Include incoming gradients of read-only inputs, output, scratch, and
        # inactive buffers. Aliased fields receive the same final seed in both.
        for args in (serial_args, parallel_args)
            for (i, a) in enumerate(_threaded_test_bars(args))
                a .= reshape(range(-0.03i, 0.07i; length=length(a)), size(a))
            end
        end
        expected = serial_back(Mooncake.NoRData())
        got = parallel_back(Mooncake.NoRData())
        for (a, b) in zip(_threaded_test_bars(parallel_args), _threaded_test_bars(serial_args))
            @test all(isapprox.(a, b; rtol=1e-8, atol=1e-8))
        end
        for i in (6, 10, 11, 12)
            @test got[i+1] ≈ expected[i+2] rtol=1e-8 atol=1e-8
        end
        if feedback
            @test got[9] == expected[10] == 0.0
        end
        for (a, old) in zip(_threaded_test_arrays(p), before)
            @test a == old
        end
        println("THREADED_BOUNDARY_PASSED threads=$(Threads.nthreads()) feedback=$feedback alias_inputs=$alias_inputs alias_omitted=$alias_omitted use_fast_specials=$use_fast_specials")
    end
    bad = _threaded_boundary_fixture()
    bad[4].W2_buf[1] = bad[4].W_buf[1]
    @test_throws ArgumentError Mooncake.rrule!!(
        Mooncake.zero_fcodual(Mapse.HMcode._assemble_threaded_slices!),
        _threaded_test_coduals(bad)...)
    bad = _threaded_boundary_fixture()
    bad[4].W_buf[2] = bad[4].W_buf[1]
    @test_throws ArgumentError Mooncake.rrule!!(
        Mooncake.zero_fcodual(Mapse.HMcode._assemble_threaded_slices!),
        _threaded_test_coduals(bad)...)
    p = _threaded_boundary_fixture()
    empty_args = (p[1], p[2], Float64[], p[4:end]...)
    cds = _threaded_test_coduals(empty_args)
    before = copy.(_threaded_test_arrays(empty_args))
    bars = _threaded_test_bars(cds)
    foreach(a -> fill!(a, 0.7), bars)
    _, back = Mooncake.rrule!!(
        Mooncake.zero_fcodual(Mapse.HMcode._assemble_threaded_slices!), cds...)
    back(Mooncake.NoRData())
    for (a, old) in zip(_threaded_test_arrays(empty_args), before)
        @test a == old
    end
    @test all(a -> all(==(0.7), a), bars)
end

@testset "Concurrent independent threaded reverse states: four callers, fifty repeats" begin
    p = _threaded_boundary_fixture(; feedback=true)
    expected = _independent_threaded_boundary(deepcopy(p))
    results = _concurrent_threaded_boundaries(ntuple(_ -> deepcopy(p), 4); repeats=50)
    for result in results
        @test result == expected
    end
    @test expected[3]
end

@testset "Unsupported storage headers reject before mutation" begin
    for unused_field in (false, true)
        p = _threaded_boundary_fixture(; feedback=true)
        ws = p[4]
        if unused_field
            # A distinct Vector header over an active input's storage. R is not
            # used by this chunk, but the full caller-graph guard still covers it.
            ws.R = vec(reshape(ws.M, :, 1))
            @test ws.R !== ws.M && Base.mightalias(ws.R, ws.M)
        else
            storage = copy(vec(ws.Pk_lin_mat))
            ws.Pk_lin_mat = reshape(storage, size(ws.Pk_lin_mat))
            ws.Pk_wig_mat = reshape(storage, size(ws.Pk_wig_mat))
            @test ws.Pk_lin_mat !== ws.Pk_wig_mat
            @test Base.mightalias(ws.Pk_lin_mat, ws.Pk_wig_mat)
        end
        cds = _threaded_test_coduals(p)
        before = copy.(_threaded_test_arrays(p))
        bars = _threaded_test_bars(cds)
        foreach(a -> fill!(a, 0.7), bars)
        @test_throws ArgumentError Mooncake.rrule!!(
            Mooncake.zero_fcodual(Mapse.HMcode._assemble_threaded_slices!), cds...)
        @test all(a == b for (a,b) in zip(_threaded_test_arrays(p), before))
        @test all(a -> all(==(0.7), a), bars)
    end
end

@testset "Output shape rejects before mutation" begin
    original = _threaded_boundary_fixture()
    for shape in ((length(original[2])+1, length(original[3])),
                  (length(original[2]), length(original[3])+1))
        p = (fill(7.3, shape), original[2:end]...)
        cds = _threaded_test_coduals(p)
        before = copy.(_threaded_test_arrays(p))
        bars = _threaded_test_bars(cds)
        foreach(a -> fill!(a, 0.7), bars)
        @test_throws DimensionMismatch Mooncake.rrule!!(
            Mooncake.zero_fcodual(Mapse.HMcode._assemble_threaded_slices!), cds...)
        @test all(a == b for (a,b) in zip(_threaded_test_arrays(p), before))
        @test all(a -> all(==(0.7), a), bars)
    end
end

@testset "Storage traversal handles cycles, closures and undefined entries" begin
    ext = Base.get_extension(Mapse, :MapseMooncakeExt)
    values = [1.0,2.0,3.0,4.0]
    captured = let values=values
        () -> values
    end
    container = Vector{Any}(undef, 4)
    container[1], container[2], container[3] = container, captured, Ref{Any}()
    arrays = AbstractArray[]
    ext._threaded_array_leaves!(arrays, IdDict{Any,Nothing}(), container)
    @test count(a -> a === container, arrays) == 1
    @test count(a -> a === values, arrays) == 1
    @test length(arrays) == 2

    @static if isdefined(Base, :Memory)
        memory = Memory{Any}(undef, 4)
        alias = vec(reshape(values, 2, 2))
        @test alias !== values && Base.mightalias(alias, values)
        memory[1], memory[2], memory[3] = memory, captured, alias
        empty!(arrays)
        ext._threaded_array_leaves!(arrays, IdDict{Any,Nothing}(), memory)
        @test count(a -> a === memory, arrays) == 1
        @test count(a -> a === values, arrays) == 1
        @test count(a -> a === alias, arrays) == 1
        @test length(arrays) == 3
        # Direct numeric Memory aliases also participate in the overlap guard.
        backing = getfield(getfield(values, :ref), :mem)
        empty!(arrays)
        ext._threaded_array_leaves!(arrays, IdDict{Any,Nothing}(), (values, backing))
        @test length(arrays) == 2
        @test Base.mightalias(arrays[1], arrays[2])
    end
end
