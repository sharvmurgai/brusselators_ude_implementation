# ---------------------------------------------------------------------------
# SciML UDE Brusselator - Debug + Training Mode (FBDF, N=16, Float32)
# ---------------------------------------------------------------------------

using Pkg
Pkg.activate("ude_project")

using DifferentialEquations, Lux, ComponentArrays, Random, Plots
using Optimization, OptimizationOptimJL, SciMLSensitivity, Zygote, OrdinaryDiffEq

# --- 1. Constants & Grid Setup ---
N_GRID = 16
XYD = range(0f0, stop = 1f0, length = N_GRID)
dx = step(XYD)
T_FINAL = 11.5f0
SAVE_AT = 0.5f0
tspan = (0.0f0, T_FINAL)
t_points = range(tspan[1], stop=tspan[2], step=SAVE_AT)
A, B, alpha = 3.4f0, 1.0f0, 10.0f0

brusselator_f(x, y, t) = (((x - 0.3f0)^2 + (y - 0.6f0)^2) <= 0.01f0) * (t >= 1.1f0) * 5.0f0
limit(a, N) = a == 0 ? N : a == N+1 ? 1 : a

function init_brusselator(xyd)
    println("[Init] Creating initial condition array...")
    u0 = zeros(Float32, N_GRID, N_GRID, 2)
    for I in CartesianIndices((N_GRID, N_GRID))
        x, y = xyd[I[1]], xyd[I[2]]
        u0[I,1] = 22f0 * (y * (1f0 - y))^(3f0/2f0)
        u0[I,2] = 27f0 * (x * (1f0 - x))^(3f0/2f0)
    end
    println("[Init] Done.")
    return u0
end
u0 = init_brusselator(XYD)

# --- 2. Ground Truth PDE ---
function pde_truth!(du, u, p, t)
    A, B, alpha, dx = p
    αdx = alpha / dx^2
    for I in CartesianIndices((N_GRID, N_GRID))
        i, j = Tuple(I)
        x, y = XYD[i], XYD[j]
        ip1, im1 = limit(i+1, N_GRID), limit(i-1, N_GRID)
        jp1, jm1 = limit(j+1, N_GRID), limit(j-1, N_GRID)
        U, V = u[i,j,1], u[i,j,2]
        ΔU = u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4f0 * U
        ΔV = u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4f0 * V
        du[i,j,1] = αdx*ΔU + B + U^2 * V - (A+1f0)*U + brusselator_f(x, y, t)
        du[i,j,2] = αdx*ΔV + A*U - U^2 * V
    end
end

p_tuple = (A, B, alpha, dx)
println("[Ground Truth Solver - DEBUG] Solving using FBDF()...")
@time sol_truth = solve(ODEProblem(pde_truth!, u0, tspan, p_tuple), FBDF(), saveat=t_points)
u_true = Array(sol_truth)
println("[Ground Truth] Success. Shape: ", size(u_true))

# --- 3. Define UDE (NN replaces U^2*V) ---
println("[UDE Setup] Building neural network...")
model = Lux.Chain(Dense(2 => 16, tanh), Dense(16 => 1))
rng = Random.default_rng()
ps_init, st = Lux.setup(rng, model)
ps_init = ComponentArray(ps_init)

function pde_ude!(du, u, ps_nn, t)
    αdx = alpha / dx^2
    for I in CartesianIndices((N_GRID, N_GRID))
        i, j = Tuple(I)
        x, y = XYD[i], XYD[j]
        ip1, im1 = limit(i+1, N_GRID), limit(i-1, N_GRID)
        jp1, jm1 = limit(j+1, N_GRID), limit(j-1, N_GRID)
        U, V = u[i,j,1], u[i,j,2]
        ΔU = u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4f0 * U
        ΔV = u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4f0 * V
        nn_val, _ = model([U, V], ps_nn, st)
        val = nn_val[1]
        du[i,j,1] = αdx*ΔU + B + val - (A+1f0)*U + brusselator_f(x, y, t)
        du[i,j,2] = αdx*ΔV + A*U - val
    end
end

prob_ude_template = ODEProblem(pde_ude!, u0, tspan, ps_init)

# --- 4. Loss Function ---
println("[Loss] Defining loss function...")
function loss_fn(ps, _)
    println("  > Solving UDE with new parameters...")
    prob = remake(prob_ude_template, p=ps)
    sol = solve(prob, FBDF(), saveat=t_points, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    if sol.retcode != ReturnCode.Success
        println("  > Failed solve: ", sol.retcode)
        return Inf32
    end
    pred = Array(sol)
    lval = sum(abs2, pred .- u_true) / length(u_true)
    println("  > Solve OK. Loss = ", lval)
    return lval
end

# --- 5. Optimization ---
println("[Training] Starting optimization...")
using OptimizationOptimisers
optf = OptimizationFunction(loss_fn, AutoZygote())
optprob = OptimizationProblem(optf, ps_init)
loss_history = Float32[]

callback = (ps, l) -> begin
    push!(loss_history, l)
    println("Epoch $(length(loss_history)): Loss = $l")
    false
end

res = solve(optprob, Optimisers.Adam(0.01), callback=callback, maxiters=10)

# --- 6. Plot Final Comparison ---
println("[Plot] Final U/V comparison plots...")
center = N_GRID ÷ 2
sol_final = solve(remake(prob_ude_template, p=res.u), FBDF(), saveat=t_points)
pred = Array(sol_final)

p1 = plot(t_points, u_true[center,center,1,:], lw=2, label="U True")
plot!(p1, t_points, pred[center,center,1,:], lw=2, ls=:dash, label="U Pred")
title!(p1, "Center U Concentration Over Time")

p2 = plot(t_points, u_true[center,center,2,:], lw=2, label="V True")
plot!(p2, t_points, pred[center,center,2,:], lw=2, ls=:dash, label="V Pred")
title!(p2, "Center V Concentration Over Time")

plot(p1, p2, layout=(1,2), size=(900,400))
