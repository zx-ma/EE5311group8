# 0 angle is pointing upwards
import jax
import jax.numpy as jnp

from pendulum.physics import step


def pid_control(state, error_sum, params):
    angle, angular_v = state
    Kp, Ki, Kd = params
    error = 0 - angle
    torque = Kp * error + Ki * error_sum + Kd * (0 - angular_v)
    return torque


def simulate_pid(state0, pid_params, dt, n_steps, env_params):
    def scan_fn(carry, _):
        state, error_sum = carry
        error = 0 - state[0]
        torque = pid_control(state, error_sum, pid_params)
        next_state = step(state, torque, dt, env_params)
        next_error_sum = error_sum + error * dt
        return (next_state, next_error_sum), (next_state, torque)

    init_carry = (state0, 0.0)
    _, (traj, torques) = jax.lax.scan(scan_fn, init_carry, jnp.arange(n_steps))
    return traj, torques
