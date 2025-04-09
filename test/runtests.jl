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

effort_emu = Mapse.LinearPkEmulator(TrainedEmulator = emu, kgrid=k_test, zgrid=z_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Postprocessing = postprocessing)

x = [Ωcb0, h, mν, w0, wa]

n = 64
x1 = vcat([0.], sort(rand(n-2)), [1.])
x2 = 2 .* vcat([0.], sort(rand(n-2)), [1.])
y = rand(n)

W = rand(2, 20, 3, 10)
v = rand(20, 10)

function di_spline(y,x,xn)
    spline = QuadraticSpline(y,x; extrapolation = ExtrapolationType.Extension)
    return spline.(xn)
end

function D_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse._D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function f_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse._f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function r_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse._r_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function r_z_check_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse._r_z_check(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

myx = Array(LinRange(0., 1., 100))
monotest = sin.(myx)
quadtest = 0.5.*cos.(myx)
hexatest = 0.1.*cos.(2 .* myx)
q_par = 1.4
q_perp = 0.6

x3 = Array(LinRange(-1., 1., 100))

@testset "Mapse tests" begin
    @test isapprox(Mapse._E_a(1, Ωcb0, h), 1.)
    #@test isapprox(Zygote.gradient(x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-5)
    #@test isapprox(grad(central_fdm(5,1), x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-3)
    #@test isapprox(Zygote.gradient(x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-5)
    #@test isapprox(grad(central_fdm(5,1), x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-4)
    #@test isapprox(grad(central_fdm(5,1), x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-7)
    #@test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-6)
    #@test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], Zygote.gradient(x->r_z_check_x(3., x), x)[1], rtol=1e-7)
    @test isapprox(Mapse._r_z(3., Ωcb0, h; mν =mν, w0=w0, wa=wa), Mapse._r_z_check(3., Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-6)
    @test isapprox(Mapse._r_z(10., 0.14/0.67^2, 0.67; mν =0.4, w0=-1.9, wa=0.7), 10161.232807937273, rtol=2e-4)
    D, f = Mapse._D_f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa)
    @test isapprox(D, Mapse._D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
    @test isapprox(f, Mapse._f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
    @test isapprox([Mapse._f_z(myz, Ωcb0, h; mν =mν, w0=w0, wa=wa) for myz in z],  Mapse._f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-10)
    @test isapprox([Mapse._D_z(myz, Ωcb0, h; mν =mν, w0=w0, wa=wa) for myz in z],  Mapse._D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-10)
    mycosmo = Mapse.w0waCDMCosmology(ln10Aₛ=3., nₛ=0.96, h=0.636, ωb=0.02237, ωc = 0.1, mν=0.06, w0=-2., wa=1.)
    mycosmo_ref = Mapse.w0waCDMCosmology(ln10Aₛ=3., nₛ=0.96, h=0.6736, ωb=0.02237, ωc = 0.12, mν=0.06, w0=-1., wa=0.)
end
