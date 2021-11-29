import gym
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def evaluate_model_naive(model, num_simulations=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_simulations: (int) number of simulations to evaluate it
    :return: (float) Mean reward for the last num_simulations
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_rewards = []
    for i in range(num_simulations):
        simulation_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            simulation_rewards.append(reward)

        all_rewards.append(sum(simulation_rewards))

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f'Mean reward: {mean_reward} +-{std_reward}; Num Simulations: {num_simulations}')
    return mean_reward


def record_video(env_id, model, video_length=1000, eval_env=None, prefix='', video_folder='Out/Videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    if eval_env is None:
        eval_env = DummyVecEnv([lambda: gym.make(env_id)])

    eval_env = VecVideoRecorder(
        eval_env, video_folder=video_folder,
        record_video_trigger=lambda step: step == 0, video_length=video_length,
        name_prefix=prefix
    )

    obs = eval_env.reset()
    total_reward = 0
    for time_step in range(video_length):
        action, _state = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
    print(f'video total reward: {total_reward}')

    eval_env.close()
