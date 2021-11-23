# import nevergrad
import numpy as np

from games.tiling_no_render import TilePlacingEnv

if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])


    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[1] = -1.0  # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[1] = 0


    # for i in range(3):
    #     env = TilePlacing()
    #     env.reset()
    #     print(f'Track len: {len(env.track)}')
    #     # print(env.track[:, 1])
    #     print(f'World: {env.world_state.shape}')
    #     print(f'{env.world_state[:10]}')
    # print(env.road)
    # exit()
    env = TilePlacingEnv()
    env.max_steps_without_reward = 5000
    env.max_steps = 100000
    env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
