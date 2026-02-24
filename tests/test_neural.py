import sys
from pathlib import Path

import jax.numpy as jnp

from controllers.neural import load_policy, nn_loss, save_policy, simulate_nn, train_nn
from pendulum.physics import DEFAULT_PARAMS
from pendulum.visualize import animate_cartpole, plot_trajectory

state0 = jnp.array([0.0, 0.1, 0.0, 0.0])
dt = 0.01
horizon = 800
eval_horizon = 1000
model_path = Path("data") / "neural" / "policy.eqx"

if ("--load" in sys.argv or "-l" in sys.argv) and model_path.exists():
    print("--- loading saved policy ---")
    policy = load_policy(model_path)
else:
    print("--- training neural controller ---")
    policy = train_nn(state0, dt, horizon, DEFAULT_PARAMS, lr=0.001, epochs=2600)
    save_policy(policy, model_path)

loss = nn_loss(policy, state0, dt, horizon, DEFAULT_PARAMS)
print(f"\nfinal loss: {float(loss):.6f}")

# evaluate on longer horizon
traj, forces = simulate_nn(policy, state0, dt, eval_horizon, DEFAULT_PARAMS)
print(f"final theta: {float(traj[-1, 1]):.5f}")
print(f"final x:     {float(traj[-1, 0]):.5f}")

plot_trajectory(traj, dt)
animate_cartpole(traj, dt, save_path=Path("data") / "neural" / "neural.gif")

print("\nneural test done")
