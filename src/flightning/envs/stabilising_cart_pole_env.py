import functools
from typing import Optional

import chex
import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from flightning.objects import CartPole, CartPoleState
from flightning.utils import spaces
import flightning.envs.env_base as env_base
from flightning.envs.env_base import EnvTransition


@jdc.pytree_dataclass
class EnvState(env_base.EnvState):
    time: float
    step_idx: int
    cart_pole_state: CartPoleState


class StabilisingCartPoleEnv(env_base.Env[EnvState]):
    """Cart-pole stabilization environment."""

    def __init__(
        self,
        *,
        max_steps_in_episode=500,
        dt=0.02,
        # Cart-pole parameters
        gravity=9.81,
        length=1.0,
        mass_cart=1.0,
        mass_pole=0.1,
        friction_cart=0.0,
        friction_pole=0.0,
        # Initial state randomisation
        x_std=0.1,
        theta_std=0.1,
        x_dot_std=0.05,
        theta_dot_std=0.05,
        # Cost weights
        q_x=1.0,  # position penalty
        q_theta=10.0,  # angle penalty (more important)
        q_x_dot=0.1,  # velocity penalty
        q_theta_dot=0.1,  # angular velocity penalty
        r_force=0.01,  # control effort penalty
        # Constraints
        x_limit=2.4,  # cart position limit
        theta_limit=jnp.pi / 6,  # angle limit (30 degrees)
        force_limit=10.0,  # maximum force
    ):
        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)

        # Cart-pole model
        self.cart_pole = CartPole(
            gravity=gravity,
            length=length,
            mass_cart=mass_cart,
            mass_pole=mass_pole,
            friction_cart=friction_cart,
            friction_pole=friction_pole,
        )

        # Initial state randomization parameters
        self.x_std = x_std
        self.theta_std = theta_std
        self.x_dot_std = x_dot_std
        self.theta_dot_std = theta_dot_std

        # LQR cost weights
        self.q_x = q_x
        self.q_theta = q_theta
        self.q_x_dot = q_x_dot
        self.q_theta_dot = q_theta_dot
        self.r_force = r_force

        # State and action limits
        self.x_limit = x_limit
        self.theta_limit = theta_limit
        self.force_limit = force_limit

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, state: Optional[EnvState] = None
    ) -> tuple[EnvState, jax.Array]:
        """Resetto environmentto."""

        key_x, key_theta, key_x_dot, key_theta_dot = jax.random.split(key, 4)

        # Random initial state (small perturbations from equilibrium)
        # x = self.x_std * jax.random.normal(key_x)
        # theta = self.theta_std * jax.random.normal(key_theta)
        # x_dot = self.x_dot_std * jax.random.normal(key_x_dot)
        # theta_dot = self.theta_dot_std * jax.random.normal(key_theta_dot)

        # Random position and angle uniformly
        x = jax.random.uniform(
            key_x, minval=-self.x_limit, maxval=self.x_limit
        )
        theta = jax.random.uniform(
            key_theta, minval=-self.theta_limit, maxval=self.theta_limit
        )
        x_dot = self.x_dot_std * jax.random.normal(key_x_dot)
        theta_dot = self.theta_dot_std * jax.random.normal(key_theta_dot)

        cart_pole_state = self.cart_pole.create_state(
            x=x, theta=theta, x_dot=x_dot, theta_dot=theta_dot
        )

        # noinspection PyArgumentList
        state = EnvState(
            time=0.0,
            step_idx=0,
            cart_pole_state=cart_pole_state,
        )

        obs = self._get_obs(state)
        return state, obs

    def _get_obs(self, state: EnvState) -> jax.Array:
        """Get observation: [x, sin(theta), cos(theta), x_dot, theta_dot].
        *** We use continuous representation for angle theta. ***
        """
        s = state.cart_pole_state
        return jnp.array([
            s.x,
            jnp.sin(s.theta),
            jnp.cos(s.theta),
            s.x_dot,
            s.theta_dot,
        ])

    # Core functions
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: EnvState, action: jax.Array, key: chex.PRNGKey
    ) -> EnvTransition:
        """Step the environment forward by one timestep."""

        # Clip action to limits
        force = jnp.clip(action[0], -self.force_limit, self.force_limit)

        # Simulate cart-pole dynamics
        cart_pole_state = self.cart_pole.step(
            state.cart_pole_state, force, self.dt
        )

        # Update state
        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            cart_pole_state=cart_pole_state,
        )

        obs = self._get_obs(next_state)
        reward = self._get_reward(next_state, force)
        terminated = self._is_terminal(next_state)
        truncated = jnp.greater_equal(
            next_state.step_idx, self.max_steps_in_episode
        )

        return EnvTransition(
            next_state, obs, reward, terminated, truncated, dict()
        )

    def _get_reward(self, state: EnvState, force: jax.Array) -> jax.Array:
        """Conpute reward (negative loss function)."""
        s = state.cart_pole_state

        # Quadratic state costs
        x_cost = self.q_x * s.x ** 2
        theta_cost = self.q_theta * s.theta ** 2
        x_dot_cost = self.q_x_dot * s.x_dot ** 2
        theta_dot_cost = self.q_theta_dot * s.theta_dot ** 2

        # Quadratic control cost
        force_cost = self.r_force * force ** 2

        # Total cost
        cost = x_cost + theta_cost + x_dot_cost + theta_dot_cost + force_cost

        # Return negative cost as reward (scaled by dt)
        reward = -self.dt * cost
        return reward

    def _is_terminal(self, state: EnvState) -> jax.Array:
        """Episode terminates if cart goes out of bounds."""
        s = state.cart_pole_state

        # Check if cart is out of position limits
        cart_out = jnp.abs(s.x) > self.x_limit

        # Check if pole has fallen (angle too large)
        # pole_fallen = jnp.abs(s.theta) > self.theta_limit

        # Terminal if either condition is true
        # is_terminal = jnp.logical_or(cart_out, pole_fallen)
        is_terminal = cart_out
        return is_terminal

    @property
    def action_space(self) -> spaces.Box:
        """Action is a single force value."""
        low = jnp.array([-self.force_limit])
        high = jnp.array([self.force_limit])
        return spaces.Box(low, high, shape=(1,))

    @property
    def observation_space(self) -> spaces.Box:
        """Observation is [x, sin(theta), cos(theta), x_dot, theta_dot]."""
        # Approximate velocity bounds (used for normalisation)
        v_max = 10.0
        omega_max = 10.0

        low = jnp.array([
            -self.x_limit,
            -1.0,  # sin(theta)
            -1.0,  # cos(theta)
            -v_max,
            -omega_max,
        ])
        high = jnp.array([
            self.x_limit,
            1.0,  # sin(theta)
            1.0,  # cos(theta)
            v_max,
            omega_max,
        ])
        return spaces.Box(low, high, shape=(5,))

    @property
    def hovering_action(self) -> jax.Array:
        """Equilibrium action (no force for cart-pole)."""
        return jnp.array([0.0])

    def plot_trajectories(self, traj: EnvTransition):
        """Plot cart-pole trajectories."""
        assert traj.reward.ndim == 2
        num_trajs = traj.reward.shape[0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        state: EnvState = traj.state
        done = np.logical_or(traj.terminated, traj.truncated)

        for i in range(num_trajs):
            # Find episode length
            idx = np.where(done[i])[0]
            if len(idx) > 0:
                idx = idx[0].item() + 1
            else:
                idx = len(done[i])

            # Extract trajectory
            t = state.time[i, :idx]
            x = state.cart_pole_state.x[i, :idx]
            theta = state.cart_pole_state.theta[i, :idx]
            x_dot = state.cart_pole_state.x_dot[i, :idx]
            theta_dot = state.cart_pole_state.theta_dot[i, :idx]

            # Plot position
            axes[0, 0].plot(t, x, alpha=0.7)
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Cart Position (m)")
            axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[0, 0].grid(True, alpha=0.3)

            # Plot angle
            axes[0, 1].plot(t, theta, alpha=0.7)
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Pole Angle (rad)")
            axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[0, 1].grid(True, alpha=0.3)

            # Plot cart velocity
            axes[1, 0].plot(t, x_dot, alpha=0.7)
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Cart Velocity (m/s)")
            axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].grid(True, alpha=0.3)

            # Plot pole angular velocity
            axes[1, 1].plot(t, theta_dot, alpha=0.7)
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Pole Angular Velocity (rad/s)")
            axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.suptitle(f"Cart-Pole Trajectories ({num_trajs} rollouts)", y=1.02)
        plt.show()


if __name__ == "__main__":
    from flightning.utils.random import key_generator

    key_gen = key_generator(0)

    env = StabilisingCartPoleEnv()

    state, obs = env.reset(next(key_gen))
    print("Initial observation:", obs)

    random_action = env.action_space.sample(next(key_gen))
    transition = env.step(state, random_action, next(key_gen))
    print("Next observation:", transition.obs)
    print("Reward:", transition.reward)
