from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pendulum.physics import DEFAULT_PARAMS, simulate
from pendulum.visualize import animate_pendulum, plot_trajectory

state0 = jnp.array([jnp.pi / 4, 0.0])
controls = jnp.zeros(1000)
traj = simulate(state0, controls, 0.01, DEFAULT_PARAMS)

plot_trajectory(traj, 0.01)
plt.show()


def final_angle(theta0):
    state0 = jnp.array([theta0, 0.0])
    controls = jnp.zeros(500)
    traj = simulate(state0, controls, 0.01, DEFAULT_PARAMS)
    return traj[-1, 0]


grad_fn = jax.grad(final_angle)
print(grad_fn(0.5))


animate_pendulum(traj, 0.01, save_path=Path("data") / "default" / "1.gif")
