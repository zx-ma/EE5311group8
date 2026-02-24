from pathlib import Path

import jax.numpy as jnp

from controllers.optimize_pid import optimize_pid, pid_loss
from controllers.pid import simulate_pid
from pendulum.physics import DEFAULT_PARAMS
from pendulum.visualize import animate_cartpole, plot_trajectory

state0 = jnp.array([0.0, 0.1, 0.0, 0.0])
dt = 0.01

# use shorter horizon for optimization (3 sec) longer for evaluation 10 sec
opt_steps = 300
eval_steps = 1000

# manual params from hand-tuning
manual_params = jnp.array([500.0, 30.0, 3.0, 5.0])
manual_loss = pid_loss(manual_params, state0, dt, opt_steps, DEFAULT_PARAMS)
print(f"manual params: {manual_params}")
print(f"manual loss:   {float(manual_loss):.6f}\n")

# optimize with short horizon
print("--- optimizing")
optimized_params = optimize_pid(manual_params, state0, dt, opt_steps, DEFAULT_PARAMS, lr=0.1, epochs=2000)
optimized_loss = pid_loss(optimized_params, state0, dt, opt_steps, DEFAULT_PARAMS)
print(f"\noptimized params: {optimized_params}")
print(f"optimized loss:   {float(optimized_loss):.6f}")
print(f"improvement:      {float(manual_loss - optimized_loss):.6f}")

# evaluate both on longer horizon
traj_manual, _ = simulate_pid(state0, manual_params, dt, eval_steps, DEFAULT_PARAMS)
traj_opt, _ = simulate_pid(state0, optimized_params, dt, eval_steps, DEFAULT_PARAMS)

plot_trajectory(traj_manual, dt)
plot_trajectory(traj_opt, dt)

animate_cartpole(traj_manual, dt, save_path=Path("data") / "pid" / "manual.gif")
animate_cartpole(traj_opt, dt, save_path=Path("data") / "pid" / "optimized.gif")

print("\noptimize pid test done")
