import jax.numpy as jnp

from pendulum.physics import DEFAULT_PARAMS, dynamics, simulate, step

dt = 0.01
params = DEFAULT_PARAMS


# state: [x, theta, x_dot, theta_dot]
# tilted pendulum, everything else at rest
state = jnp.array([0.0, jnp.pi / 4, 0.0, 0.0])
deriv = dynamics(state, 0.0, params)
assert deriv[0] == 0.0, "cart velocity should be zero"
assert deriv[1] == 0.0, "angular velocity should be zero"
assert deriv[3] > 0.0, "gravity should produce positive angular acceleration"
print(f"dynamics ok: deriv = {deriv}")


# step: one rk4 step should change the state
next_state = step(state, 0.0, dt, params)
assert not jnp.allclose(state, next_state), "state should change after one step"
print(f"step ok: {state} -> {next_state}")


# simulate: free cart-pole for 5 seconds
n_steps = 500
controls = jnp.zeros(n_steps)
traj = simulate(state, controls, dt, params)
assert traj.shape == (n_steps, 4), f"unexpected shape: {traj.shape}"

# pole angle should keep increasing (falls over)
angles = traj[:, 1]
assert float(angles[-1]) > float(angles[0]), "pendulum should fall away from upright"
print(f"simulate ok: {n_steps} steps, angle range [{float(angles.min()):.3f}, {float(angles.max()):.3f}]")

print("\nall good oh~yeeeee!")
