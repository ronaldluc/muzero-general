import os
import sys

import nevergrad
import numpy
import ray

from muzero import MuZero, load_model_menu, hyperparameter_search

if __name__ == "__main__":
    game_file = 'tiling'
    # Train directly with "python muzero.py cartpole"
    muzero = MuZero(game_file)
    muzero.train()
    muzero.test()
    ray.shutdown()
