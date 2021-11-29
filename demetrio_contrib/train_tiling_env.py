import gym
import numpy as np
from time import sleep
from stable_baselines3 import PPO, DQN, SAC, A2C, TD3
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy
# from stable_baselines3.common.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from pyvirtualdisplay import Display
from demetrio_contrib.tools import record_video
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from demetrio_contrib.sb_save_best_callback import SaveOnBestTrainingRewardCallback
from demetrio_contrib.plot_training_history import plot_training_history
from demetrio_contrib.ipython_autoreload import *
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


N_sim_eval = 20
num_cpu = 4
gym_base_name = 'TilePlacingEnv'
episode_length = 500
gym_name = f'{gym_base_name}_long{episode_length}-v0'
# N_steps_train = 1000000 # *10*5
N_steps_train = 100000*3*10
# N_steps_train = 10000*3
n_stack = 32
check_training_freq_steps = 10000

log_dir = 'Out/Logs/'

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
env = VecFrameStack(env, n_stack=n_stack)
env = VecMonitor(env, log_dir)
model = A2C(MlpPolicy, env, verbose=1, learning_rate=0.0003/1000, n_steps=100)



print('')
print('---- Reward Before Training ----')
print('')
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=int(N_sim_eval))
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


print('')
print('---- Training ----')
print('')
log_and_save = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
model.learn(total_timesteps=N_steps_train) #, callback=log_and_save)
plot_training_history(
    log_dir=log_dir,
    save_path=f'{log_dir}tiling_train',
    rolling_avg_window=10,
)


print('')
print('---- Recoding a Demo Video ----')
print('')
from stable_baselines3.common.vec_env import DummyVecEnv
eval_env = DummyVecEnv([lambda: gym.make(gym_name)])
eval_env = VecFrameStack(eval_env, n_stack=n_stack)
record_video(eval_env=eval_env, model=model, prefix='tiling_train', env_id=None)



print('')
print('---- Reward After Training ----')
print('')
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_sim_eval)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



