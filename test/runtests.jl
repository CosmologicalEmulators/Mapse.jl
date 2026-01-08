using Test
using NPZ
using SimpleChains
using Static
using Mapse
using DataInterpolations

mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

k_test = Array(LinRange(0,200, 40))
z_test = Array(LinRange(0, 5, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6,2)
outminmax = rand(40,2)
a, Ωcb0, mν, h, w0, wa = [1., 0.3, 0.06, 0.67, -1.1, 0.2]
#a, Ωcb0, mν, h, w0, wa = [1., 0.3, 0.06, 0.67, -1.1, 0.2]
z = Array(LinRange(0., 3., 100))

emu = Mapse.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

postprocessing = (input, output, D, Pkemu) -> output
preprocessing = (input) -> input

effort_emu = Mapse.MapseEmulator(TrainedEmulator = emu, kgrid=k_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Preprocessing = preprocessing,
                                Postprocessing = postprocessing)

pkcb_emu = Mapse.MapseEmulator(TrainedEmulator = emu, kgrid=k_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Preprocessing = preprocessing,
                                Postprocessing = postprocessing)

postprocessing_boost = (input, output, emu) -> output
boost_emu = Mapse.NonLinearBoostPkEmulator(TrainedEmulator = emu, kgrid=k_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Preprocessing = preprocessing,
                                Postprocessing = postprocessing_boost)

x = [Ωcb0, h, mν, w0, wa]

n = 64
x1 = vcat([0.], sort(rand(n-2)), [1.])
x2 = 2 .* vcat([0.], sort(rand(n-2)), [1.])
y = rand(n)

function D_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse.D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function f_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse.f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

myx = Array(LinRange(0., 1., 100))
monotest = sin.(myx)
quadtest = 0.5.*cos.(myx)
hexatest = 0.1.*cos.(2 .* myx)
q_par = 1.4
q_perp = 0.6

x3 = Array(LinRange(-1., 1., 100))

@testset "Mapse tests" begin
    @test isapprox(Mapse.E_a(1, Ωcb0, h), 1.)
    #@test isapprox(Zygote.gradient(x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-5)
    #@test isapprox(grad(central_fdm(5,1), x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-3)
    #@test isapprox(Zygote.gradient(x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-5)
    #@test isapprox(grad(central_fdm(5,1), x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-4)
    #@test isapprox(grad(central_fdm(5,1), x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-7)
    #@test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-6)
    #@test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], Zygote.gradient(x->r_z_check_x(3., x), x)[1], rtol=1e-7)
    D, f = Mapse.D_f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa)
    @test isapprox(D, Mapse.D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
    @test isapprox(f, Mapse.f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
    @test isapprox([Mapse.f_z(myz, Ωcb0, h; mν =mν, w0=w0, wa=wa) for myz in z],  Mapse.f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-10)
    @test isapprox([Mapse.D_z(myz, Ωcb0, h; mν =mν, w0=w0, wa=wa) for myz in z],  Mapse.D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-10)
    mycosmo = Mapse.w0waCDMCosmology(ln10Aₛ=3., nₛ=0.96, h=0.636, ωb=0.02237, ωc = 0.1, mν=0.06, w0=-2., wa=1.)
    mycosmo_ref = Mapse.w0waCDMCosmology(ln10Aₛ=3., nₛ=0.96, h=0.6736, ωb=0.02237, ωc = 0.12, mν=0.06, w0=-1., wa=0.)

    # Test get_Pk
    pk_scalar = Mapse.get_Pk(x, 1.0, 1.0, effort_emu)
    @test size(pk_scalar) == (40,)

    pk_vector = Mapse.get_Pk(x, [0.5, 1.0], [1.0, 1.0], effort_emu)
    @test size(pk_vector) == (40, 2)

    # Test get_Pk for Boost
    boost_scalar = Mapse.get_Pk(x, 1.0, boost_emu)
    @test size(boost_scalar) == (40,)

    boost_vector = Mapse.get_Pk(x, [0.5, 1.0], boost_emu)
    @test size(boost_vector) == (40, 2)

    # Test PkEmulator (Combined)
    full_emu = Mapse.PkEmulator(LinearPmm = effort_emu, LinearPkcb = pkcb_emu, Boost = boost_emu)

    full_pk_scalar = Mapse.get_Pk(x, 1.0, 1.0, full_emu)
    @test size(full_pk_scalar) == (40,)

    full_pk_vector = Mapse.get_Pk(x, [0.5, 1.0], [1.0, 1.0], full_emu)
    @test size(full_pk_vector) == (40, 2)

    # Test get_linear_Pmm and get_linear_Pkcb
    linear_pmm_scalar = Mapse.get_linear_Pmm(x, 1.0, 1.0, full_emu)
    @test linear_pmm_scalar == Mapse.get_Pk(x, 1.0, 1.0, effort_emu)

    linear_pkcb_vector = Mapse.get_linear_Pkcb(x, [0.5, 1.0], [1.0, 1.0], full_emu)
    @test linear_pkcb_vector == Mapse.get_Pk(x, [0.5, 1.0], [1.0, 1.0], pkcb_emu)
end
