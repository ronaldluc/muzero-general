import datetime
import math
import os
import time
from itertools import product

import Box2D
import gym
import numpy
import numpy as np
import pyglet
import torch
from Box2D.b2 import contactListener
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from gym import spaces
from gym.utils import seeding, EzPickle
from gym.envs.classic_control.rendering import make_polygon

from games.abstract_game import AbstractGame
# from abstract_game import AbstractGame
from simulation.robot_dynamics import Robot
from simulation.hover_drone import Drone
from scipy.spatial.distance import euclidean as l2_dist

pyglet.options["debug_gl"] = False
from pyglet import gl
# from skspatial.objects import Line, Points, Vector
from scipy.spatial.transform import Rotation as R

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]


# class MuZeroConfigSpecial:
#     def __init__(self):
#         # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
#
#         self.seed = 0  # Seed for numpy, torch and the game
#         self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
#
#         ### Game
#         self.env = TilePlacingEnv()
#         # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
#         self.observation_shape = (
#             1, 1, 3 * (1 + self.env.closest_track_points))  # ([angle, y, x], 1, [*car, *world])
#         # Fixed list of all possible actions. You should only edit the length
#         self.action_space = list(range(len(self.env.discrete_actions)))
#         self.players = list(range(1))  # List of players. You should only edit the length
#         self.stacked_observations = 10  # Number of previous observations and previous actions to add to the current observation
#
#         # Evaluate
#         self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
#         self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class
#
#         ### Self-Play
#         self.num_workers = 16  # Number of simultaneous threads/workers self-playing to feed the replay buffer
#         self.selfplay_on_gpu = False
#         self.max_moves = 1000  # Maximum number of moves if game is not finished before
#         self.num_simulations = 50  # Number of future moves self-simulated
#         self.discount = 0.997  # Chronological discount of the reward
#         self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
#
#         # Root prior exploration noise
#         self.root_dirichlet_alpha = 0.25
#         self.root_exploration_fraction = 0.25
#
#         # UCB formula
#         self.pb_c_base = 19652
#         self.pb_c_init = 1.25
#
#         ### Network
#         self.network = "fullyconnected"  # "resnet" / "fullyconnected"
#         self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
#
#         # Residual Network
#         self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
#         self.blocks = 2  # Number of blocks in the ResNet
#         self.channels = 32  # Number of channels in the ResNet
#         self.reduced_channels_reward = 32  # Number of channels in reward head
#         self.reduced_channels_value = 32  # Number of channels in value head
#         self.reduced_channels_policy = 32  # Number of channels in policy head
#         self.resnet_fc_reward_layers = [
#             16]  # Define the hidden layers in the reward head of the dynamic network
#         self.resnet_fc_value_layers = [
#             16]  # Define the hidden layers in the value head of the prediction network
#         self.resnet_fc_policy_layers = [
#             16]  # Define the hidden layers in the policy head of the prediction network
#
#         # Fully Connected Network
#         self.encoding_size = 32
#         self.fc_representation_layers = [
#             16]  # Define the hidden layers in the representation network
#         self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
#         self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
#         self.fc_value_layers = [16]  # Define the hidden layers in the value network
#         self.fc_policy_layers = [16]  # Define the hidden layers in the policy network
#
#         ### Training
#         self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
#                                          os.path.basename(__file__)[:-3],
#                                          datetime.datetime.now().strftime(
#                                              "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
#         self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
#         self.training_steps = 20e3  # Total number of training steps (ie weights update according to a batch)
#         self.batch_size = 512  # Number of parts of games to train on at each training step
#         self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
#         self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
#         self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
#         print("Training on GPU:", self.train_on_gpu)
#
#         self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
#         self.weight_decay = 1e-4  # L2 weights regularization
#         self.momentum = 0.9  # Used only if optimizer is SGD
#
#         # Exponential learning rate schedule
#         self.lr_init = 0.05  # Initial learning rate
#         self.lr_decay_rate = 0.75  # Set it to 1 to use a constant learning rate
#         self.lr_decay_steps = 1e3
#
#         ### Replay Buffer
#         self.replay_buffer_size = 2e3  # Number of self-play games to keep in the replay buffer
#         self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
#         self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
#         self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
#         self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
#
#         # Reanalyze (See paper appendix Reanalyse)
#         self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
#         self.reanalyse_on_gpu = False  # Use GPU for the reanalyse phase. Paper recommends false (on CPU)
#
#         ### Adjust the self play / training ratio to avoid over/underfitting
#         self.self_play_delay = 0  # Number of seconds to wait after each played game
#         self.training_delay = 0  # Number of seconds to wait after each training step
#         self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
#
#     def visit_softmax_temperature_fn(self, trained_steps):
#         """
#         Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
#         The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
#
#         Returns:
#             Positive float.
#         """
#         if trained_steps < 0.5 * self.training_steps:
#             return 1.0
#         elif trained_steps < 0.75 * self.training_steps:
#             return 0.5
#         else:
#             return 0.25


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.env = TilePlacingEnv()
        self.env.reset()
        # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = self.env.step(None)[0].shape  # ([angle, y, x], 1, [*car, *world])

        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(len(self.env.discrete_actions)))
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 32  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 15  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = self.env.max_steps  # Maximum number of moves if game is not finished before
        self.num_simulations = 30  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 196520      # 196520
        self.pb_c_init = 4.25   # 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 300  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 32  # Number of blocks in the ResNet
        self.channels = 256  # Number of channels in the ResNet
        self.reduced_channels_reward = 256  # Number of channels in reward head
        self.reduced_channels_value = 256  # Number of channels in value head
        self.reduced_channels_policy = 256  # Number of channels in policy head
        self.resnet_fc_reward_layers = [256,
                                        256]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [256,
                                       256]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [256,
                                        256]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        base = 16
        self.encoding_size = 2 * base
        self.fc_representation_layers = [base, base]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [base, base]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [base]  # Define the hidden layers in the reward network
        self.fc_value_layers = [base]  # Define the hidden layers in the value network
        self.fc_policy_layers = [base]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(
            100e3)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 1024  # Number of parts of games to train on at each training step
        self.checkpoint_interval = int(1e3)  # Number of training steps before using the model for self-playing
        self.checkpoint_interval = int(3e1)  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        # self.use_sgd()
        self.use_adam()

        ### Replay Buffer
        self.replay_buffer_size = int(
            1e6)  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False  # torch.cuda.is_available()

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def use_sgd(self):
        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD
        # Exponential learning rate schedule
        self.lr_init = 0.05  # Initial learning rate
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 35e3

    def use_adam(self):
        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 5e-3  # L2 weights regularization
        self.weight_decay = 5e-5  # L2 weights regularization
        # Exponential learning rate schedule
        self.lr_init = 0.0001  # Initial learning rate
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 35e3

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 50e3:
            return 1.0
        elif trained_steps < 75e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        # self.env = gym.make("CartPole-v1")
        # self.env = TilingEnv()
        self.env = TilePlacingEnv()
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        action = self.env.discrete_actions[action]
        observation, reward, done, _ = self.env.step(action)
        # print(f'{observation.mean()}')
        return observation, reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(len(self.env.discrete_actions)))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()
        # return np.swapaxes(numpy.array(self.env.reset()), 0, 2)

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        time.sleep(0.01)
        # input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        action = self.env.discrete_actions[action_number]
        return f"left gas: {action[0]} right gas: {action[1]}"


"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.

Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                # self.env.reward += 1000.0 / len(self.env.track)   # TODO: remove
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)


class TilePlacingEnv(gym.Env, EzPickle):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(self, verbose=0):
        EzPickle.__init__(self)
        self.can_fly = False  # set to True for EZ God-Mod flying robot. Helpful for baselines.
        self.tile_visited_count = None
        self.max_steps_without_reward = 100
        self.max_steps = 1200
        self.min_checkpoint_delta = 5 # 25  # px
        self.num_future_tiles = 1
        self.reward_deteriation_per_tick = 0.0 # -0.0001  #-0.01
        self.reward_per_checkpoint = 10  # default was 1
        self.reward_per_unit_getting_closer = 1  # reward for moving in the right direction
        self.last_step_positive_reward = None
        self.last_step_dist_next_tile = None
        self.dist_next_tile = 0
        self.dist_diff = 0
        self.track_samples = 100
        self.closest_track_points = 10
        self.steps = None
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.t = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.action_space = spaces.Box(
            np.array([-1, -1, ]).astype(np.float32),
            np.array([+1, +1, ]).astype(np.float32),
        )  # left_wheel, right_wheel (back wheels)

        # ad paper: https://neuro.cs.ut.ee/wp-content/uploads/2018/02/2d_racing.pdf
        # ACTIONS = [[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.3, 0.0], [0.0, 0.0, 0.8]]

        # if you want to train a discrete model
        discrete_step = 1 / 3
        self.discrete_actions = list(product(
            np.arange(2 / discrete_step + 1) * discrete_step - 1,
            np.arange(2 / discrete_step + 1) * discrete_step - 1,
        ))

        # Vision-compatible observation space (original)
        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        # )

        # Location-based observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 1, 3*(1+self.num_future_tiles)), dtype=np.float32
        )

        self.start = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def reset(self):
        if self.verbose:
            self.print_performance()
        self._destroy()
        self.last_step_positive_reward = 0
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.highest_tile_in_seq = 0
        self.t = 0.0
        self.road_poly = []
        if self.can_fly:
            Robot_model = Drone
        else:
            Robot_model = Robot

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        # self.car = Robot(self.world, *self.track[0][1:4])
        angle_diviation = 0.1
        # self.car = Robot(self.world, np.random.normal(-angle_diviation * np.pi, angle_diviation * np.pi),
        #                  *self.track[0][2:4])  # Random angle
        dir = self.world_state[1, 1:] - self.world_state[0, 1:]
        dir /= np.linalg.norm(dir)
        angle = np.arctan2(dir[1], dir[0])
        self.car = Robot_model(self.world, np.random.normal(angle - np.pi / 2, angle_diviation * np.pi),
                         *self.track[0][2:4])  # Random angle

        self.start = time.time()
        self.steps = 0

        next_tile = self.world_state[self.highest_tile_in_seq]
        x, y = self.car.hull.position
        # dist_next_tile = np.sqrt(((next_tile[1:3] - (x, y)) ** 2).sum())
        dist_next_tile = l2_dist(next_tile[1:3], (x, y))
        self.last_step_dist_next_tile = dist_next_tile

        return self.step(None)[0]

    def print_performance(self):
        if self.t and self.start:
            print(f'{self.t / (self.start - time.time()):4.6f}x real-time \t '
                  f'{self.steps / (time.time() - self.start)} steps/s')

    def step(self, action):
        """
        # steer, gas, brake
        # [(-1, 1), (0, 1), (0, 1)]
        right, left
        [(-1, 1), (-1, 1)]
        """
        # if self.steps == 500:
        #     self.print_performance()
        self.steps += 1
        if action is not None:
            # action = ACTIONS[action]
            if np.all(np.isclose(action, np.zeros_like(action), atol=1e-3)):
                self.car.brake(1)
                if self.verbose:
                    print("brake ")
            else:
                self.car.brake(0)
                self.car.gas_wheel(gas=action[0], wheel_id=2)
                self.car.gas_wheel(gas=action[1], wheel_id=3)
            # self.car.steer(-action[0])
            # self.car.gas(action[1])
            # self.car.brake(action[2])

        tick = 1.0 / FPS
        self.car.step(tick)
        self.world.Step(tick, 6 * 30, 2 * 30)
        self.t += tick

        # self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward += self.reward_deteriation_per_tick
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            # self.car.fuel_spent = 0.0

            # All tiles visited
            if self.tile_visited_count == len(self.track):
                done = True

            # Out of the field
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                self.reward += 10 * self.reward_deteriation_per_tick * (
                        self.max_steps - self.steps)
                done = True

            # Next tile
            next_tile = self.world_state[self.highest_tile_in_seq]
            # self.dist_next_tile = np.sqrt(((next_tile[1:3] - (x, y)) ** 2).sum())
            self.dist_next_tile = l2_dist(next_tile[1:3], (x, y))

            # reward for getting closer
            dist_diff = self.last_step_dist_next_tile - self.dist_next_tile
            self.dist_diff = dist_diff
            self.reward += dist_diff*self.reward_per_unit_getting_closer

            # reward for getting the checkpoint
            if self.dist_next_tile < self.min_checkpoint_delta:
                self.highest_tile_in_seq += 1
                self.reward += self.reward_per_checkpoint

                # recompute next tile
                next_tile = self.world_state[self.highest_tile_in_seq]
                # self.dist_next_tile = np.sqrt(((next_tile[1:3] - (x, y)) ** 2).sum())
                self.dist_next_tile = l2_dist(next_tile[1:3], (x, y))

            self.last_step_dist_next_tile = self.dist_next_tile

            # Does not move
            self.last_step_positive_reward = 0 if self.reward > self.prev_reward else self.last_step_positive_reward + 1
            if self.last_step_positive_reward > self.max_steps_without_reward:
                self.reward += 2 * self.reward_deteriation_per_tick * (self.max_steps - self.steps)
                done = True

            # Calculate reward change
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

        # track: _, angle, x, y
        car_state = np.array([self.car.hull.angle % (2 * np.pi),
                              self.car.hull.position.x,
                              self.car.hull.position.y])[None, :]
        # direction = R.from_euler('z', self.car.hull.angle, degrees=False).as_rotvec()
        # car_line = Line(self.car.hull.position, direction)

        distances = np.sum((self.world_state[:, 1:] - self.car.hull.position) ** 2,
                           axis=-1) ** 0.5
        closest_id = np.argsort(distances, axis=0)[:self.closest_track_points]
        near_track = self.world_state[closest_id]

        relative_near_track = near_track - car_state

        # state = np.concatenate((car_state, self.world_state), axis=-1)
        next_tile_diff = self.world_state[self.highest_tile_in_seq:self.highest_tile_in_seq + self.num_future_tiles] \
                         - car_state  # angle, x, y
        next_tile_diff[:, 1:] = next_tile_diff[:, 1:] * 10    # rescale to make more similar with car position
        # print(next_tile_diff, self.highest_tile_in_seq)

        state = np.concatenate((car_state, next_tile_diff), axis=0)
        # print(f'state: {state.shape}')
        # state[0, 1:] = 0.0  # hide car XY coordinates
        state[:, 1:] = state[:, 1:] / PLAYFIELD  # normalize 0-1 range coords
        state[:, 0] = state[:, 0] / (2 * np.pi)  # normalize angle to 0-1 range
        # state = np.concatenate((next_tile_diff, ), axis=-1).flatten()
        self.state = state.flatten()[None, None, :]  # muzero compatibility
        # print(f'{step_reward:5.4f} | {self.state}')
        # print(next_tile_diff[1:])
        # print(f'car_state: {car_state.shape} word_state: {self.world_state.shape}')
        # print(f'state: {self.state.shape} ')

        return self.state, step_reward, done, {}  # gym env compatible return

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()

        # render check-points
        def point_to_polygon_box(point_xy, inflate_by=1, color_rgb=[1, 0, 0]):
            left_bottom = point_xy + [-inflate_by, -inflate_by]
            left_top = point_xy + [-inflate_by, +inflate_by]
            right_top = point_xy + [+inflate_by, +inflate_by]
            right_bottom = point_xy + [+inflate_by, -inflate_by]
            box = make_polygon([left_bottom, left_top, right_top, right_bottom])
            box.set_color(*color_rgb)
            return box

        for state_ in self.world_state:
            checkpoint_box = point_to_polygon_box(point_xy=state_[1:3], inflate_by=1, color_rgb=[0.5, 0, 0])
            checkpoint_box.render()

        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            distances = np.sum((self.world_state[:, 1:] - self.car.hull.position) ** 2, axis=-1) ** 0.5
            closest_id = np.argmin(distances, axis=0)
            next_tile_diff = self.world_state[self.highest_tile_in_seq:self.highest_tile_in_seq + 2,
                             1:] - self.car.hull.position  # angle, x, y
            print(f'Closest: {closest_id:5} {distances[closest_id]:6.4f} '
                  f'should go for {self.highest_tile_in_seq:5} {distances[self.highest_tile_in_seq]:6.4f} \n'
                  # f'{next_tile_diff[:]}'
                  f'{self.state}')
            assert self.car.hull.position == (self.car.hull.position.x, self.car.hull.position.y)
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                    track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1: i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )

        track = np.array(track)  # shape: (n_points, [alpha, beta, x, y])
        self.track = track
        self.world_state = np.array([np.interp(np.arange(0, 1, 1 / self.track_samples) * len(t),
                                               np.arange(0, len(t)),
                                               t) for t in track.T[1:]]).T  # shape: (samples, [beta, x, y])
        return True

    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        k = PLAYFIELD / 20.0
        colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()
        if np.isnan(self.reward):
            self.score_label.text = 'NA'
        else:
            self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


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
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0


    env = TilePlacingEnv()
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
