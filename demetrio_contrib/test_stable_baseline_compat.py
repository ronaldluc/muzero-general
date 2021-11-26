import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from pyvirtualdisplay import Display

gym.envs.register(
    id='test-v0',
    entry_point=f'games.tiling_no_render:TilePlacingEnv',
)

_display = Display(visible=False, size=(1400, 900))
_ = _display.start()


def record_video(env_id, video_length=1000, prefix='', video_folder='Out/Videos/'):
    """
    :param env_id: (str)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env, video_folder=video_folder,
        record_video_trigger=lambda step: step == 0, video_length=video_length,
        name_prefix=prefix
    )

    obs = eval_env.reset()
    total_reward = 0
    for time_step in range(video_length):
        action = [env.action_space.sample()]
        # action, _state = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
    print(f'video total reward: {total_reward}')

    # Close the video recorder
    eval_env.close()

record_video(env_id='test-v0', prefix='test_gym')
