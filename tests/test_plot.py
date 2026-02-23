from pathlib import Path

import jax
import jax.numpy as jnp

from pendulum.physics import DEFAULT_PARAMS, simulate
from pendulum.visualize import animate_cartpole, plot_trajectory

# free falling cartpole
state0 = jnp.array([0.0, jnp.pi / 4, 0.0, 0.0])
controls = jnp.zeros(1000)
traj = simulate(state0, controls, 0.01, DEFAULT_PARAMS)

plot_trajectory(traj, 0.01)


def final_angle(theta0):
    state0 = jnp.array([0.0, theta0, 0.0, 0.0])
    controls = jnp.zeros(500)
    traj = simulate(state0, controls, 0.01, DEFAULT_PARAMS)
    return traj[-1, 1]


grad_fn = jax.grad(final_angle)
print(grad_fn(0.5))

animate_cartpole(traj, 0.01, save_path=Path("data") / "default" / "cartpole_free.gif")
