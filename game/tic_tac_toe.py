import datetime
import os
import gym
import numpy as np
import torch
from .abstract_game import AbstractGame
from .pygame_display import Pygame_Tictactoe


class MuZeroConfig:
    def __init__(self):
        self.seed = 0

        ###Game
        self.observation_shape = (3, 3, 3)  # Dimensions of the game observation
        self.action_space = [i for i in range(9)]  # Posible actions list
        self.players = [i for i in range(2)]
        self.stacked_observations = 0

        ###Evaluate
        self.muzero_player = 0  # Define turn Muzero
        self.opponent = "expert"

        # Self-play
        self.num_actors = 1  # Simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 9  # Maximum number of moves before finishing
        self.num_simulations = 25  # Future moves self-simulated
        self.discount = 1  # Chronological discount reward
        self.temperature_threshold = 6  # Moves before dropping temperature to 0

        # Root prior exploration noise
        self.root_dirichlet_alpha = .1
        self.root_exploration_fraction = .25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "resnet"
        self.support_size = 10

        # Residual Network
        self.downsample = False  # Downsample observations before representation network
        self.blocks = 1  # Blocks in the ResNet
        self.channels = 16  # Channels in the ResNet
        self.reduced_channels = 16  # Channels before heads of dynamic and prediction networks
        self.resnet_fc_reward_layers = [8]  # Hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Hidden layers in the reward network
        self.fc_value_layers = []  # Hidden layers in the value network
        self.fc_policy_layers = []  # Hidden layers in the policy network

        # Training
        self.results_path = os.path.join(os.path.dirname(__file__),
                                         "../results",
                                         os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Store the model weights
        self.training_steps = 100000
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "Nadam"  # "Adam" or "SGD".
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        # Replay Buffer
        self.window_size = 3000  # Self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Game moves to keep for every batch element
        self.td_steps = 20  # Steps in the future to take into account for calculating the target value
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # Use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Seconds to wait after each played game
        self.training_delay = 0  # Seconds to wait after each training step
        self.ratio = None  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution. Ensure that the action selection becomes greedier.
        :param trained_steps:
        :return:
        """
        return 1


class TicTactToe(object):
    def __init__(self):
        self.board = np.zeros((3, 3)).astype(int)
        self.player = 1

    def get_current_player(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = np.zeros((3, 3)).astype(int)
        self.player = 1
        return self.get_observation()

    def perform_step(self, action):
        row = action // 3
        col = action % 3
        self.board[row, col] = self.player
        done = self.have_winner() or len(self.get_legal_actions()) == 0
        reward = 1 if self.have_winner() else 0
        self.player *= -1
        print(self.get_observation(), reward, done)
        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = np.where(self.board == 1, 1., 0.)
        board_player2 = np.where(self.board == -1, 1., 0.)
        board_to_play = np.full((3, 3), self.player).astype(float)
        return np.array([board_player1, board_player2, board_to_play])

    def get_legal_actions(self):
        legal_actions = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal_actions.append(i)
        return legal_actions

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * np.ones(3).astype(int)).all():
                return True
            if (self.board[:, i] == self.player * np.ones(3).astype(int)).all():
                return True

        # Diagonal checks
        if (
                self.board[0, 0] == self.player and
                self.board[1, 1] == self.player and
                self.board[2, 2] == self.player
        ):
            return True
        if (
                self.board[2, 0] == self.player and
                self.board[1, 1] == self.player and
                self.board[0, 2] == self.player
        ):
            return True

        return False

    def expert_action(self):
        board = self.board
        action = np.random.choice(self.get_legal_actions())
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = np.where(board[i, :] == 0)[0][0]
                action = np.ravel_multi_index((np.array([i]), np.array([ind])), (3, 3))[0]
                if self.player * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = np.where(board[:, i] == 0)[0][0]
                action = np.ravel_multi_index((np.array([ind]), np.array([i])), (3, 3))[0]
                if self.player * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = np.where(diag == 0)[0][0]
            action = np.ravel_multi_index((np.array([ind]), np.array([ind])), (3, 3))[0]
            if self.player * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = np.where(anti_diag == 0)[0][0]
            action = np.ravel_multi_index((np.array([ind]), np.array([2 - ind])), (3, 3))[0]
            if self.player * sum(anti_diag) > 0:
                return action

        return action

    def render(self):
        Pygame_Tictactoe.draw_board(self.board[::-1])
        # CAMBIARRRR....
        # print(self.board)
        # print(self.board[::-1])


class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = TicTactToe()

    def perform_step(self, action):
        observation, reward, done = self.env.perform_step(action)
        return observation, reward * 20, done

    def get_legal_actions(self):
        return self.env.get_legal_actions()

    def render(self):
        self.env.render()
        input("Press to take a step")

    def get_action_expert_agent(self):
        pass

    def get_current_player(self):
        return self.env.get_current_player()

    def close(self):
        pass

    @property
    def get_human_action(self):
        while True:
            try:
                row = int(
                    input(
                        "Enter the row (1, 2 or 3) to play for the player {}: ".format(
                            self.get_current_player()
                        )
                    )
                )
                col = int(
                    input(
                        "Enter the column (1, 2 or 3) to play for the player {}: ".format(
                            self.get_current_player()
                        )
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if choice in self.get_legal_actions() and 1 <= row <= 3 and 1 <= col <= 3:
                    break

            except:
                pass
            print("Wrong input, try again")
        return choice

    def expert_action(self):
        return self.env.expert_action()

    def convert_action_to_string(self, action_number):
        row = 3 - action_number // 3
        col = action_number % 3 + 1
        return f"Play row {col}, column{row} "

    def reset(self):
        return self.env.reset()
