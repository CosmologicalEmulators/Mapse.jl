using Capse
using BenchmarkTools
using SimpleChains
using JSON

"""
mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 4999)
)

weights = rand(500000)
ℓgrid = ones(2000)
InMinMax_array = zeros(6,2)
OutMinMax_array = zeros(4999,2)
path_json = "./src/assets/nn_setup.json"
nn_setup = JSON.parsefile(path_json)
emu = Capse.SimpleChainsEmulator(Architecture = mlpd, Weights = weights,
                                 Description = nn_setup)
postprocessing(input, output, Cℓemu) = output .* exp(input[1]-3.)
Cℓ_emu = Capse.CℓEmulator(TrainedEmulator = emu, ℓgrid=ℓgrid, InMinMax = InMinMax_array,
                                OutMinMax = OutMinMax_array,
                                Postprocessing = postprocessing)"""
Cℓ_emu_SC = Capse.load_emulator("TT/")
Cℓ_emu_Lux = Capse.load_emulator("TT/", emu = Capse.LuxEmulator)


suite = BenchmarkGroup()

suite["Capse"] = BenchmarkGroup(["tag1"])
input_test = rand(6)
suite["Capse"]["SimpleChains"] = @benchmarkable Capse.get_Cℓ($input_test, $Cℓ_emu_SC)
suite["Capse"]["Lux"] = @benchmarkable Capse.get_Cℓ($input_test, $Cℓ_emu_Lux)

tune!(suite)

@benchmark Capse.get_Cℓ($input_test, $Cℓ_emu_SC)
@benchmark Capse.get_Cℓ($input_test, $Cℓ_emu_Lux)
a = @benchmark Capse.get_Cℓ($input_test, $Cℓ_emu_SC)
b = @benchmark Capse.get_Cℓ($input_test, $Cℓ_emu_Lux)
println(a)
println(b)

results = run(suite, verbose = true)

BenchmarkTools.save("./src/assets/capse_benchmark.json", results)
