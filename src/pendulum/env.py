import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pendulum.physics import DEFAULT_PARAMS


def _dynamics_np(state, force, env_params):
    x, theta, x_dot, theta_dot = state
    g = env_params["g"]
    pole_len = env_params["l"]
    mc = env_params["mc"]
    mp = env_params["mp"]

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    denom = mc + mp * sin_t**2

    x_ddot = (mp * pole_len * theta_dot**2 * sin_t - mp * g * sin_t * cos_t + force) / denom
    theta_ddot = ((mc + mp) * g * sin_t - mp * pole_len * theta_dot**2 * sin_t * cos_t - force * cos_t) / (
        pole_len * denom
    )
    return np.array([x_dot, theta_dot, x_ddot, theta_ddot])


def _step_np(state, force, dt, env_params):
    k1 = _dynamics_np(state, force, env_params)
    k2 = _dynamics_np(state + 0.5 * dt * k1, force, env_params)
    k3 = _dynamics_np(state + 0.5 * dt * k2, force, env_params)
    k4 = _dynamics_np(state + dt * k3, force, env_params)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class CartPoleEnv(gym.Env):
    def __init__(self, dt=0.01, max_steps=1000, env_params=None):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.env_params = env_params or DEFAULT_PARAMS

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-200.0, high=200.0, shape=(1,), dtype=np.float32)

        self.state = np.zeros(4)
        self.step_count = 0

    def _get_obs(self):
        x, theta, x_dot, theta_dot = self.state
        return np.array([x, np.sin(theta), np.cos(theta), x_dot, theta_dot], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        theta0 = self.np_random.uniform(-0.1, 0.1)
        self.state = np.array([0.0, theta0, 0.0, 0.0])
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        force = float(np.clip(action[0], -200.0, 200.0))
        self.state = _step_np(self.state, force, self.dt, self.env_params)
        self.step_count += 1

        theta = self.state[1]
        x = self.state[0]
        theta_dot = self.state[3]

        reward = -((1 - np.cos(theta)) ** 2 + 0.1 * x**2 + 0.1 * theta_dot**2 + 0.001 * force**2)

        terminated = bool(abs(x) > 5.0 or abs(theta) > np.pi / 2)
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {}
