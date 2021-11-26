import gym


gym.envs.register(
    id='test-v0',
    entry_point=f'games.tiling_no_render:TilePlacingEnv',
)

from pyvirtualdisplay import Display
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

env = gym.make('test-v0')
observation = env.reset()
total_reward = 0
done = False
while not done:
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        observation = env.reset()
print(total_reward)
env.close()
