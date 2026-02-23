import jax
import jax.numpy as jnp

from pendulum.physics import step


def pid_control(state, error_sum, params):
    x, theta, x_dot, theta_dot = state
    Kp_theta, Kd_theta, Kp_x, Kd_x = params

    # positive force pushes cart right, which tilts pole left (reduces positive theta)
    force = Kp_theta * theta + Kd_theta * theta_dot + Kp_x * (0 - x) + Kd_x * (0 - x_dot)
    return force


def simulate_pid(state0, pid_params, dt, n_steps, env_params):
    def scan_fn(carry, _):
        state, error_sum = carry
        force = pid_control(state, error_sum, pid_params)
        next_state = step(state, force, dt, env_params)
        next_error_sum = error_sum + state[1] * dt
        return (next_state, next_error_sum), (next_state, force)

    init_carry = (state0, 0.0)
    _, (traj, forces) = jax.lax.scan(scan_fn, init_carry, jnp.arange(n_steps))
    return traj, forces
