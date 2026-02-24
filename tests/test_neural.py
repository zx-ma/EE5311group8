from pathlib import Path

import jax.numpy as jnp

from controllers.neural import nn_loss, simulate_nn, train_nn
from pendulum.physics import DEFAULT_PARAMS
from pendulum.visualize import animate_cartpole, plot_trajectory

state0 = jnp.array([0.0, 0.1, 0.0, 0.0])
dt = 0.01
horizon = 800
eval_horizon = 1000

print("--- training neural controller ---")
policy = train_nn(state0, dt, horizon, DEFAULT_PARAMS, lr=0.001, epochs=2000)

loss = nn_loss(policy, state0, dt, horizon, DEFAULT_PARAMS)
print(f"\nfinal loss: {float(loss):.6f}")

# evaluate on longer horizon
traj, forces = simulate_nn(policy, state0, dt, eval_horizon, DEFAULT_PARAMS)
print(f"final theta: {float(traj[-1, 1]):.5f}")
print(f"final x:     {float(traj[-1, 0]):.5f}")

plot_trajectory(traj, dt)
animate_cartpole(traj, dt, save_path=Path("data") / "neural" / "neural.gif")

print("\nneural test done")
