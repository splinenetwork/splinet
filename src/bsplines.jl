#=  Evaluating all the non-zero B_k,p(x) for a specific x
    p:  degree of the spline
    k:  indicating left boundary of support of B_k,p(x)
         B_k,p(x) is non-zero only in [t_k, t_k+p+1]
    Bx: a (p+1)-dim vector, [B_k-p,p(x); ...; B_k,p(x)]
=#
struct BSpline{T,D}
    Bx::Vector{T}
end
(BSpline{T,D}() where {T,D}) = BSpline{T,D}(zeros(T,D+1))

function bspline_coefficients_update!(bsp::BSpline{T,D}, k::Int, x::T, δ::T) where {T,D}
    fill!(bsp.Bx, T(0.0))
    bsp.Bx[1] = T(1.0)
    for i in 1:D
        for r in i:-1:1
            α1 = (x - (k-i+r)*δ) / ((k+r)*δ - (k-i+r)*δ)
            α2 = ((k+r+1)*δ - x) / ((k+r+1)*δ - (k-i+r+1)*δ)
            bsp.Bx[r+1] = α1 * bsp.Bx[r] + α2 * bsp.Bx[r+1]
        end
        bsp.Bx[1] = bsp.Bx[1] * ((k+1)*δ - x) / ((k+1)*δ - (k-i+1)*δ)
    end
end
