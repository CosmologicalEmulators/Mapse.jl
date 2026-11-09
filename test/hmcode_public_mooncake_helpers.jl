function _hmcode_public_mooncake_problem(; fast=false, feedback=false, threaded=false,
                                       fixed_redshifts=nothing, fixed_temperature=nothing)
    k = collect(10.0 .^ range(-4, 2; length=64))
    zc = fixed_redshifts === nothing ?
         (fast ? collect(range(0.0, 1.0; length=5)) : [0.0, 0.5, 1.0]) :
         copy(fixed_redshifts)
    z = fast ? collect(range(0.0, 1.0; length=9)) : copy(zc)
    pk = [1e4*(kv/0.1)^0.96/(1+(kv/0.1)^2)^2/(1+zv)^2 for kv in k, zv in zc]
    weights = reshape(collect(range(0.7, 1.3; length=length(k)*length(z))), length(k), :)
    objective = function (x)
        cosmo = Mapse.HMCodeCosmology(x[1:9]...)
        mm, cb = pk .* exp(x[10]), pk .* exp(x[10]) .* 1.01
        temperature = feedback ?
            (fixed_temperature === nothing ? 10.0^x[11] : fixed_temperature) : nothing
        result = if fast
            Mapse.hmcode_pmm_fast(cosmo, zc, z, k, mm; pk_cb_coarse=cb,
                T_AGN=temperature, nM=8, threaded=threaded)
        else
            Mapse.hmcode_pmm(cosmo, z, k, mm; pk_cb_z=cb,
                T_AGN=temperature, nM=8, threaded=threaded)
        end
        return sum(weights .* log.(result)) / length(result)
    end
    return objective, [0.31, 0.049, 0.67, 0.965, 0.81, -0.93, 0.12, 0.0014, 0.01, 0.0, 7.8]
end
