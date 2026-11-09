using Mapse, DifferentiationInterface, ForwardDiff, FiniteDiff, Interpolations
const H = Mapse.HMcode

# Optional reference-generation diagnostic; see test/README.md for execution
# in Pkg.test's sandbox. It prints evidence and never overwrites fixtures.
# Numerical reference, not an independent physics model: same equation,
# initial condition and spline as production, with a controlled ODE tolerance.
function reference_growth(c; lcdm, tolerance)
    bg = (Omega_m=c.Omega_m, Omega_k=lcdm ? 0.0 : c.Omega_k,
          w0=lcdm ? -1.0 : c.w0, wa=lcdm ? 0.0 : c.wa)
    a0 = 1e-4
    fraction = 1.0 - H._Omega_m_a(a0, bg; LCDM=lcdm)
    u0 = [a0^(1.0 - 3.0 * fraction / 5.0),
          (1.0 - 3.0 * fraction / 5.0) * a0^(-3.0 * fraction / 5.0)]
    function rhs!(du, u, parameters, a)
        background, flat = parameters
        fv = -(2.0 + H._AH(a, background; LCDM=flat) /
                     H._Hubble2(a, background; LCDM=flat)) * u[2] / a
        fd = 1.5 * H._Omega_m_a(a, background; LCDM=flat) * u[1] / a^2
        du[1] = u[2]
        du[2] = fv + fd
    end
    problem = H.ODEProblem(rhs!, u0, (a0, 1.0), (bg, lcdm))
    solution = H.solve(problem, H.Tsit5(); reltol=tolerance, abstol=tolerance)
    nodes = range(a0, 1.0; length=129)
    values = [solution(a)[1] for a in nodes]
    return extrapolate(scale(interpolate(values, BSpline(Cubic(Line(OnGrid())))), nodes), Line())
end

function main()
    c = Mapse.HMCodeCosmology(0.3, 0.05, 0.6736, 0.96, 0.8, -1.0, 0.0, 0.001, 0.0)
    names = fieldnames(typeof(c))
    a = [0.1, 0.5, 1.0]
    ad = AutoForwardDiff()
    default_fd = AutoFiniteDiff(; fdtype=Val(:central))
    for lcdm in (false, true), name in (:Omega_m, :w0, :wa, :Omega_k)
        lcdm && name != :Omega_m && continue
        index = findfirst(==(name), names)
        cosmology(x) = Mapse.HMCodeCosmology(ntuple(
            i -> i == index ? x[1] : getfield(c, i), length(names))...)
        production(x) = sum(H.get_growth_interpolator(cosmology(x); LCDM=lcdm), a)
        point = [getfield(c, name)]
        ap = prepare_gradient(production, ad, point)
        fp = prepare_gradient(production, default_fd, point)
        av, ag = value_and_gradient(production, ap, ad, point)
        fg = gradient(production, fp, default_fd, point)
        println("PRODUCTION field=", name, " LCDM=", lcdm, " value=", av,
            " AD=", ag[1], " default_FD=", fg[1])
        flush(stdout)
        for tolerance in (1e-8, 1e-11, 1e-13)
            reference(x) = sum(reference_growth(cosmology(x); lcdm, tolerance), a)
            rp = prepare_gradient(reference, ad, point)
            rv, rg = value_and_gradient(reference, rp, ad, point)
            tolerance == 1e-8 && @assert isapprox(reference(point), production(point); rtol=1e-12)
            println("REFERENCE field=", name, " LCDM=", lcdm,
                " tolerance=", tolerance, " value=", rv, " AD=", rg[1])
            for step in (1e-3, 5e-4, 1e-4, 3e-5, 1e-5, 3e-6)
                backend = AutoFiniteDiff(; fdtype=Val(:central), relstep=0.0, absstep=step)
                prep = prepare_gradient(reference, backend, point)
                g = gradient(reference, prep, backend, point)
                println("STEP field=", name, " LCDM=", lcdm,
                    " tolerance=", tolerance, " step=", step,
                    " FD=", g[1], " delta_to_AD=", g[1]-rg[1])
            end
            flush(stdout)
        end
    end
    println("GROWTH_REFERENCE_COMPLETE")
end
main()
