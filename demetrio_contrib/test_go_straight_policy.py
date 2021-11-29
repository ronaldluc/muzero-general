import gym
import numpy as np
from time import sleep
from stable_baselines3 import PPO, DQN, SAC, A2C
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy
# from stable_baselines3.common.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from pyvirtualdisplay import Display
from demetrio_contrib.tools import record_video
from demetrio_contrib.ipython_autoreload import *
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


N_sim_eval = 20
num_cpu = 4
gym_base_name = 'TilePlacingEnv'
episode_length = 500
gym_name = f'{gym_base_name}_long{episode_length}-v0'


# this will prevent spawning windows
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

gym.envs.register(
    id=gym_name,
    entry_point=f'games.tiling_no_render:TilePlacingEnv',
    max_episode_steps=episode_length,
    # reward_threshold=-110.0,
)
# env_test = gym.make(gym_name)

env = make_vec_env(
    env_id=gym_name, n_envs=num_cpu,
    env_kwargs={
        'verbose': 0,
        # 'reward_deteriation_per_tick': -0.0, #-1e4,
    }
)


print('')
print('---- Recoding a Demo Video ----')
print('')
from stable_baselines3.common.vec_env import DummyVecEnv
eval_env = DummyVecEnv([lambda: gym.make(gym_name)])
# eval_env = VecFrameStack(eval_env, n_stack=32)

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Optional, Tuple
# class GoStraight(OnPolicyAlgorithm):
class GoStraight:
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return np.array([[0.1, 0.1]]), None

go_straight = GoStraight()
record_video(eval_env=eval_env, model=go_straight, prefix='tiling_go_straight', env_id=None)


print('')
print('---- Reward GoStraight Policy ----')
print('')
mean_reward, std_reward = evaluate_policy(go_straight, eval_env, n_eval_episodes=N_sim_eval)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")