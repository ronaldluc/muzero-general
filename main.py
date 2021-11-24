# import nevergrad
import sys
from glob import glob
from time import sleep

import ray

from muzero import MuZero

if __name__ == "__main__":
    game_name = 'tiling_no_render'
    # Train directly with "python muzero.py cartpole"
    wait_time = 0
    if len(sys.argv) > 1:
        wait_time = float(sys.argv[1])
    muzero = MuZero(game_name)
    # muzero.train()
    options = ["Specify paths manually"] + sorted(glob(f"results/{game_name}/*/"))
    options.reverse()
    checkpoint_path = f"{options[0]}model.checkpoint"
    replay_buffer_path = f"{options[0]}replay_buffer.pkl"
    while True:
        muzero.load_model(
            checkpoint_path=checkpoint_path, replay_buffer_path=replay_buffer_path,
        )
        muzero.test()
        sleep(wait_time)
    ray.shutdown()