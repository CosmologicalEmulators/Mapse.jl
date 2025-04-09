function primordial_Pk(As, ns, k)
    return @. As * (k/0.05)^(ns-1)/k^3
end

function ΔM(H0, k)
    c0 = 2.99792458E8
    return k^2/H0^2*c0^2
end
