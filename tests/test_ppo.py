import os

os.environ["JAX_PLATFORMS"] = "cpu"

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pendulum.env import CartPoleEnv
from pendulum.visualize import animate_cartpole, plot_trajectory

model_path = Path("data") / "ppo" / "ppo_cartpole.zip"
model_path.parent.mkdir(parents=True, exist_ok=True)

if ("--load" in sys.argv or "-l" in sys.argv) and model_path.exists():
    print("load")
    model = PPO.load(model_path)
else:
    print("train")
    env = DummyVecEnv([lambda: CartPoleEnv()])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4, device="cpu")
    model.learn(total_timesteps=500_000)
    model.save(model_path)
    print(f"  saved to {model_path}")

# evaluate
eval_env = CartPoleEnv()
obs, _ = eval_env.reset(seed=42)
eval_env.state = np.array([0.0, 0.1, 0.0, 0.0])
obs = eval_env._get_obs()

states = [eval_env.state.copy()]
forces = []
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    states.append(eval_env.state.copy())
    forces.append(float(action[0]))
    if terminated or truncated:
        break

traj = np.array(states)
print(f"survived {len(traj)} steps")
print(f"final theta: {traj[-1, 1]:.5f}")
print(f"final x:     {traj[-1, 0]:.5f}")

plot_trajectory(traj, 0.01)
animate_cartpole(traj, 0.01, save_path=Path("data") / "ppo" / "ppo.gif")

print("\nppo test done")
