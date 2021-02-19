## Plot the result in [0,xEnd] with a uniform grid
using Plots

function spline_test(bsp::BSpline{T,D}; xEnd=1.0, nKnots::Int=6) where {T,D}
    δ = xEnd/(nKnots-1)

    n = 100
    xgrid = range(0.0, stop=xEnd, length = n+1)
    Splines = zeros(T,nKnots+D,n)

    for i in 1:n
        k = floor(Int, xgrid[i] / δ)
        bspline_coefficients_update!(bsp, k, xgrid[i], δ)
        Splines[k+1:k+D+1,i] = bsp.Bx
    end

    plot()
    for i in 1:nKnots+D
        plot!(xgrid[1:n], Splines[i,:], legend=false)
    end
    current()
end

deg = 4
bsp = BSpline{Float64,deg}()
spline_test(bsp; xEnd=1.0, nKnots=6)
