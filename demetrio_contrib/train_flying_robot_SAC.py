import os
import sys
sys.path.append('./')
os.makedirs('./Out/Logs', exist_ok=True)
os.makedirs('./Out/Videos', exist_ok=True)

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from pyvirtualdisplay import Display
from demetrio_contrib.tools import record_video
from stable_baselines3.common.noise import NormalActionNoise
from demetrio_contrib.sb_save_best_callback import SaveOnBestTrainingRewardCallback
from demetrio_contrib.plot_training_history import plot_training_history
from stable_baselines3.common.vec_env import DummyVecEnv


N_sim_eval = 20
episode_length = 2000
N_steps_train = 100000*5
n_stack = 2
check_training_freq_steps = 10000  # turned off below
gym_base_name = 'TilePlacingEnv'
gym_name = f'{gym_base_name}_long{episode_length}-v0'

log_dir = 'Out/Logs/'

# this will prevent spawning windows
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

gym.envs.register(
    id=gym_name,
    entry_point=f'games.tiling_no_render:TilePlacingEnv',
    max_episode_steps=episode_length,
)
env = DummyVecEnv([lambda: gym.make(gym_name, verbose=0, can_fly=True)])
env = VecFrameStack(env, n_stack=n_stack)
env = VecMonitor(env, log_dir)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

model = SAC(
    'MlpPolicy',
    env, verbose=1,
    learning_rate=1e-4,
    train_freq=10,
    learning_starts=10000,
    action_noise=action_noise,
)
# should work with all defaults as well

print('')
print('---- Reward Before Training ----')
print('')
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=int(N_sim_eval/10))
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


print('')
print('---- Training ----')
print('')
log_and_save = SaveOnBestTrainingRewardCallback(check_freq=check_training_freq_steps, log_dir=log_dir)
model.learn(total_timesteps=N_steps_train) #, callback=log_and_save)
plot_training_history(
    log_dir=log_dir,
    save_path=f'{log_dir}tiling_train_flying',
    rolling_avg_window=10,
)


print('')
print('---- Recoding a Demo Video ----')
print('')
from stable_baselines3.common.vec_env import DummyVecEnv
eval_env = DummyVecEnv([lambda: gym.make(gym_name, verbose=1, can_fly=True)])
eval_env = VecFrameStack(eval_env, n_stack=n_stack)
record_video(eval_env=eval_env, model=model, prefix='tiling_train_flying', env_id=None, video_length=2000)



print('')
print('---- Reward After Training ----')
print('')
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_sim_eval)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
