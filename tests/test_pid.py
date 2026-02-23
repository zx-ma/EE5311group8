from pathlib import Path

import jax.numpy as jnp

from controllers.pid import simulate_pid
from pendulum.physics import DEFAULT_PARAMS
from pendulum.visualize import animate_cartpole, plot_trajectory

# state: [x, theta, x_dot, theta_dot]
state0 = jnp.array([0.0, 0.1, 0.0, 0.0])
pid_params = jnp.array([100.0, 20.0, 5.0, 10.0])  # Kp_theta, Kd_theta, Kp_x, Kd_x
dt = 0.01
n_steps = 1000

traj, forces = simulate_pid(state0, pid_params, dt, n_steps, DEFAULT_PARAMS)

# angle should converge to 0
final_angle = float(traj[-1, 1])
print(f"final angle: {final_angle:.6f}")
assert abs(final_angle) < 0.05, f"angle should converge to 0, got {final_angle}"

# cart should not drift too far
final_x = float(traj[-1, 0])
print(f"final cart position: {final_x:.6f}")

plot_trajectory(traj, dt)
animate_cartpole(traj, dt, save_path=Path("data") / "pid" / "cartpole_pid.gif")

print("\npid test gooooood")
