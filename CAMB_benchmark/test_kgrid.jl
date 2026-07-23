using Mapse
artifact_root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
pmm_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_mm"))
println("length(get_kgrid): ", length(Mapse.get_kgrid(pmm_emu)))
