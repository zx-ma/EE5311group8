from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from controllers.pid import simulate_pid
from pendulum.physics import DEFAULT_PARAMS
from pendulum.visualize import animate_pendulum, plot_trajectory

state0 = jnp.array([0.3, 0.0])
pid_params = jnp.array([20.0, 3.0, 5.0])  # Kp, Ki, Kd
dt = 0.01
n_steps = 500

traj, torques = simulate_pid(state0, pid_params, dt, n_steps, DEFAULT_PARAMS)

# angle should converge to 0
final_angle = float(traj[-1, 0])
assert abs(final_angle) < 0.05, f"angle should converge 0, our final angle: {final_angle}"
print(f"final angle: {final_angle:.6f}")

# plot trajectory
plot_trajectory(traj, dt)
animate_pendulum(traj, dt, save_path=Path("data") / "pid" / "try_pid.gif")
plt.show()

print("\npid test gooooood")
