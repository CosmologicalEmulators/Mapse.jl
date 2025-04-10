function primordial_Pk(As, ns, k)
    c0 = 2.99792458E8
    return @. As * (k/0.05)^(ns-1)/k^3 * (k^2*c0^2)^2
end
