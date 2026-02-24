import jax.numpy as jnp

from controllers.optimize_pid import optimize_pid, pid_loss
from controllers.optimize_pid_lbfgs import optimize_pid_lbfgs
from controllers.pid import simulate_pid
from pendulum.physics import DEFAULT_PARAMS
from pendulum.visualize import animate_cartpole, plot_trajectory

state0 = jnp.array([0.0, 0.1, 0.0, 0.0])
dt = 0.01
opt_steps = 300
eval_steps = 1000

manual_params = jnp.array([500.0, 30.0, 3.0, 5.0])
manual_loss = pid_loss(manual_params, state0, dt, opt_steps, DEFAULT_PARAMS)
print(f"manual  | loss {float(manual_loss):.6f} | params {manual_params}\n")

# adam
print("--- adam (300 epochs) ")
adam_params = optimize_pid(manual_params, state0, dt, opt_steps, DEFAULT_PARAMS, lr=0.001, epochs=300)
adam_loss = pid_loss(adam_params, state0, dt, opt_steps, DEFAULT_PARAMS)
print(f"\nadam    | loss {float(adam_loss):.6f} | params {adam_params}\n")

# lbfgs
print("--- lbfgs (50 iters)")
lbfgs_params = optimize_pid_lbfgs(manual_params, state0, dt, opt_steps, DEFAULT_PARAMS, maxiter=50)
lbfgs_loss = pid_loss(lbfgs_params, state0, dt, opt_steps, DEFAULT_PARAMS)
print(f"\nlbfgs   | loss {float(lbfgs_loss):.6f} | params {lbfgs_params}\n")

# summary
print("=== summary ")
print(f"manual  | loss {float(manual_loss):.6f}")
print(f"adam    | loss {float(adam_loss):.6f}")
print(f"lbfgs   | loss {float(lbfgs_loss):.6f}")

# evaluate all three on longer horizon
for name, params in [("manual", manual_params), ("adam", adam_params), ("lbfgs", lbfgs_params)]:
    traj, _ = simulate_pid(state0, params, dt, eval_steps, DEFAULT_PARAMS)
    fig = plot_trajectory(traj, dt)
    fig.suptitle(name)
    fig.savefig(f"data/compare/{name}.png")
    animate_cartpole(traj, dt, save_path=f"data/compare/{name}.gif")
    print(f"{name:8s} | final theta {float(traj[-1, 1]):.5f} | final x {float(traj[-1, 0]):.5f}")

print("\ndone")
