import math
import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector
import functools
from gymnasium.spaces import Discrete, Box
from numpy import uint8
from abc import ABC
from numpy import zeros, uint8, array, hypot
from gymnasium import Env


def main():
    # Initialize the environment with your parameters
    env = both_far_from_plant_stag_in_mid(load_renderer=False)
    proposed_agent = ProposedAgent(get_player_0_position(env), 1, 1, 0, 1, 1, 0)
    agent_0_obs = env.observe("player_0")
    agent_1_obs = env.observe("player_1")
    env.render(mode="human")

    while True:
        proposed_agent_action = proposed_agent.choose_action(agent_0_obs)
        human_action = ""
        while human_action == "":
            human_action = input("Enter action (w: up, a: left, s: down, d: right, q: stand): ")
        print(f"Proposed agent action: {proposed_agent_action}, Human action: {human_action}")
        observation, reward, done, info = env.step(
            {'player_0': proposed_agent_action, 'player_1': wasd_to_action[human_action]})
        new_agent_obs = observation['player_0']
        print(observation)
        proposed_agent.update_parameters(agent_0_obs, new_agent_obs)
        agent_0_obs = new_agent_obs
        env.render(mode="human")


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4

wasd_to_action = {
    "w": UP,
    "a": LEFT,
    "s": DOWN,
    "d": RIGHT,
    "q": STAND
}


def calculate_distance(location_1, location_2):
    """
    Calculate the distance between two locations
    """
    location_1 = np.array(location_1)
    location_2 = np.array(location_2)
    location_1 = location_1.astype('int32')
    location_2 = location_2.astype('int32')
    return math.sqrt(math.pow((location_1[0] - location_2[0]), 2) + math.pow((location_1[1] - location_2[1]), 2))
    # return abs((location_1[0]) - location_2[0]) + abs(location_1[1] - location_2[1])


def negative_distance_delta(old_location, new_location, target_position):
    """
    Calculate the change in distance between the old location and the target position and the new location and the target position
    :param old_location: list of x and y coordinates
    :param new_location: list of x and y coordinates
    :param target_position: list of x and y coordinates
    :return: the change in distance
    """
    old_distance = calculate_distance(old_location, target_position)
    new_distance = calculate_distance(new_location, target_position)
    # if user stays on the stag, there should be a positive reward
    if old_distance == 0 and new_distance == 0:
        return 1
    return -(new_distance - old_distance)


def unpack_observation(observation):
    """
    Unpack the observation into the coordinates of the self, other player, stag, and plants
    :param observation: list of coordinates
    :return: the coordinates of the self, other player, stag, and plants
    """
    observation_int32 = observation.astype('int32')  # sometimes the observation is uint8, which causes overflow
    self_location = observation_int32[0:2]
    other_player_location = observation_int32[2:4]
    stag_location = observation_int32[4:6]
    plant_location_1 = observation_int32[6:8]
    plant_location_2 = observation_int32[8:10]
    return self_location, other_player_location, stag_location, plant_location_1, plant_location_2


def normalize_deltas(stag_weight_delta, plant_weight_delta, player_weight_delta):
    # print(
    #     f"stag_weight_delta: {stag_weight_delta}, plant_weight_delta: {plant_weight_delta}, player_weight_delta: {player_weight_delta}")
    total_delta = abs(stag_weight_delta) + abs(plant_weight_delta) + abs(player_weight_delta)
    if total_delta != 0:
        return stag_weight_delta / total_delta, plant_weight_delta / total_delta, player_weight_delta / total_delta
    else:
        # Set the deltas to zero if the total delta is zero
        return 0, 0, 0


class ProposedAgent:
    """
    An agent to play the stag hunt game, which only considers its own location as the state
    have parameters of how much it want to approach the stag, plant and the other player
    the parameters are updated based on the other player's actions
    """

    def __init__(self, location, stag_weight=1, plant_weight=1, player_weight=1,
                 stag_leanring_rate=1, plant_learning_rate=1, player_learning_rate=1):
        self.location = location
        self.stag_weight = stag_weight
        self.plant_weight = plant_weight
        self.player_weight = player_weight
        self.stag_learning_rate = stag_leanring_rate
        self.plant_learning_rate = plant_learning_rate
        self.player_learning_rate = player_learning_rate

    def normalize_weights(self):
        total_weight = self.stag_weight + self.plant_weight + self.player_weight
        if total_weight != 0:
            self.stag_weight /= total_weight
            self.plant_weight /= total_weight
            self.player_weight /= total_weight
        else:
            # Set the weights to default values if the total weight is zero
            self.stag_weight = 1
            self.plant_weight = 1
            self.player_weight = 1

    def update_parameters(self, old_obs, new_obs):
        # observation is a list of coordinates of self, other player, stag, and plants
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            old_obs)

        new_self_location, new_other_player_location, new_stag_location, new_plant_location_1, new_plant_location_2 = unpack_observation(
            new_obs)

        # Calculate the change in each weight
        stag_weight_delta = negative_distance_delta(other_player_location, new_other_player_location,
                                                    stag_location) * self.stag_learning_rate
        # Focus on the plant that is closer to the other player
        closest_plant_location = plant_location_1 if calculate_distance(plant_location_1,
                                                                        new_other_player_location) < calculate_distance(
            plant_location_2, new_other_player_location) else plant_location_2
        plant_weight_delta = negative_distance_delta(other_player_location, new_other_player_location,
                                                     closest_plant_location) * self.plant_learning_rate
        player_weight_delta = negative_distance_delta(other_player_location, new_other_player_location,
                                                      self_location) * self.player_learning_rate

        # Normalize the deltas
        stag_weight_delta, plant_weight_delta, player_weight_delta = normalize_deltas(stag_weight_delta,
                                                                                      plant_weight_delta,
                                                                                      player_weight_delta)

        # Apply the changes to the weights
        self.stag_weight += stag_weight_delta
        self.plant_weight += plant_weight_delta
        self.player_weight += player_weight_delta

        # Normalize the weights after updating them
        self.normalize_weights()

        # update the location
        self.location = new_self_location

    def calculate_action_reward(self, new_location, stag_location, plant_location_1, plant_location_2,
                                other_player_location):
        # calculate the reward for moving to a new location based on the distance to the stag, plant and the other player
        return (self.stag_weight * -calculate_distance(new_location, stag_location)
                + self.plant_weight * max(-calculate_distance(new_location, plant_location_1),
                                          -calculate_distance(new_location, plant_location_2))
                + self.player_weight * -calculate_distance(new_location, other_player_location))

    def choose_action(self, observation):
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            observation)

        # print(f"weights: stag: {self.stag_weight}, plant: {self.plant_weight}, player: {self.player_weight}")

        # get the expected reward for each action
        up_reward = self.calculate_action_reward([self_location[0], self_location[1] - 1], stag_location,
                                                 plant_location_1, plant_location_2, other_player_location)
        down_reward = self.calculate_action_reward([self_location[0], self_location[1] + 1],
                                                   stag_location, plant_location_1, plant_location_2,
                                                   other_player_location)
        left_reward = self.calculate_action_reward([self_location[0] - 1, self_location[1]],
                                                   stag_location, plant_location_1, plant_location_2,
                                                   other_player_location)
        right_reward = self.calculate_action_reward([self_location[0] + 1, self_location[1]],
                                                    stag_location, plant_location_1, plant_location_2,
                                                    other_player_location)
        still_reward = self.calculate_action_reward(self_location, stag_location, plant_location_1, plant_location_2,
                                                    other_player_location)

        # print(
        #     f"Up reward: {up_reward}, Down reward: {down_reward}, Left reward: {left_reward}, Right reward: {right_reward}, Still reward: {still_reward}")

        # choose the action with the highest expected reward
        return [left_reward, down_reward, right_reward, up_reward, still_reward].index(
            max(left_reward, down_reward, right_reward, up_reward, still_reward))


def get_player_0_position(env):
    """
    Get the position of player 0 in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :return: (x, y) coordinates
    """
    return env.env.game.A_AGENT


def set_stag_coord(env, x, y):
    """
    Set the position of the stag in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param x: x-coordinate
    :param y: y-coordinate
    """
    env.env.game.STAG = np.array([x, y])


def disable_movement_for_stag(env):
    """
    Disable the movement of the stag in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    """
    env.env.game._move_stag = lambda: True


def set_plant_positions(env, plant_positions):
    """
    Set the positions of the plant in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param plant_positions: list of (x, y) tuples
    """
    env.env.game.PLANTS = [np.array(pos) for pos in plant_positions]


def set_player_0_position(env, x, y):
    """
    Set the position of player 0 in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param x: x-coordinate
    :param y: y-coordinate
    """
    env.env.game.A_AGENT = np.array([x, y])


def set_player_1_position(env, x, y):
    """
    Set the position of player 1 in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param x: x-coordinate
    :param y: y-coordinate
    """
    env.env.game.B_AGENT = np.array([x, y])


def get_basic_env(grid_size=10, load_renderer=False):
    env = ZooHuntEnvironment(
        obs_type="coords",
        enable_multiagent=True,
        stag_follows=False,
        run_away_after_maul=False,
        forage_quantity=2,
        stag_reward=3,
        forage_reward=2,
        mauling_punishment=-1,
        load_renderer=load_renderer,
        grid_size=(grid_size, grid_size),
        respawn_plants=False,
        respawn_stag=False,
        move_closer_reward=True,
    )
    env.reset()
    return env


def both_close_to_stag(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 4, 5)
    set_player_1_position(env, 6, 5)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 1), (9, 8)])
    return env


def both_close_to_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 1), (9, 8)])
    return env


def we_close_to_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 1), (0, 8)])
    return env


def they_close_to_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(9, 1), (9, 8)])
    return env


def both_far_from_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 4, 4)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(9, 0), (0, 9)])
    return env


def stag_on_the_side_plant_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 9, 0)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(4, 4), (0, 9)])
    return env


def choose_from_stag_or_plant(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 2, 2)
    set_player_1_position(env, 7, 7)
    set_stag_coord(env, 4, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 0), (9, 9)])
    return env


class PettingZooEnv(ParallelEnv):
    def __init__(self, og_env):
        super().__init__()

        self.env = og_env

        self.possible_agents = ["player_" + str(n) for n in range(2)]
        self.agents = self.possible_agents[:]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)

        self._action_spaces = {
            agent: self.env.action_space for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: self.env.observation_space for agent in self.possible_agents
        }

        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_observations = {
            agent: self.env.observation_space.sample() for agent in self.agents
        }
        self.t = 0
        self.last_rewards = [0.0, 0.0]

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.env.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.env.action_space

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def reset(self):
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        obs = self.env.reset()
        self.accumulated_actions = []
        # Edited: fix agent 0 and 1 observing the agent coordinates in the same order
        # they should see their own coordinates first
        self.current_observations = {self.agents[0]: obs, self.agents[1]: np.concatenate((obs[2:4], obs[0:2], obs[4:]))}
        self.t = 0

        return self.current_observations

    def step(self, actions):
        observations, rewards, env_done, info = self.env.step(list(actions.values()))

        obs = {self.agents[0]: observations[0], self.agents[1]: observations[1]}
        rewards = {self.agents[0]: rewards[0], self.agents[1]: rewards[1]}
        dones = {agent: env_done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return obs, rewards, dones, infos

    def observe(self, agent):
        return self.current_observations[agent]

    def state(self):
        pass


TILE_SIZE = 32

from itertools import product
from random import choice
from sys import stdout

from numpy import all, full, zeros, uint8

symbol_dict = {"hunt": ("S", "P"), "harvest": ("p", "P"), "escalation": "M"}

A_AGENT = 0  # base
B_AGENT = 1

STAG = 2  # hunt
PLANT = 3

Y_PLANT = 2  # harvest
M_PLANT = 3

MARK = 2  # escalation


def print_matrix(obs, game, grid_size):
    if game == "escalation":
        matrix = full((grid_size[0], grid_size[1], 3), False, dtype=bool)
    else:
        matrix = full((grid_size[0], grid_size[1], 4), False, dtype=bool)

    if game == "hunt":
        a, b, stag = (obs[0], obs[1]), (obs[2], obs[3]), (obs[4], obs[5])
        matrix[a[1]][a[0]][A_AGENT] = True
        matrix[b[1]][b[0]][B_AGENT] = True
        if stag != (255, 255):
            matrix[stag[1]][stag[0]][STAG] = True
        for i in range(6, len(obs), 2):
            plant = obs[i], obs[i + 1]
            if plant != (255, 255):
                matrix[plant[1]][plant[0]][PLANT] = True

    elif game == "harvest":
        a, b = (obs[0], obs[1]), (obs[2], obs[3])
        matrix[a[0]][a[1]][A_AGENT] = True
        matrix[b[0]][b[1]][B_AGENT] = True

        for i in range(4, len(obs), 3):
            plant_age = M_PLANT if obs[i + 2] else Y_PLANT
            matrix[obs[i]][obs[i + 1]][plant_age] = True

    elif game == "escalation":
        a, b, mark = (obs[0], obs[1]), (obs[2], obs[3]), (obs[4], obs[5])
        matrix[a[0]][a[1]][A_AGENT] = True
        matrix[b[0]][b[1]][B_AGENT] = True
        matrix[mark[0]][mark[1]][MARK] = True

    symbols = symbol_dict[game]

    stdout.write("╔═" + "═════" * grid_size[0] + "══╗\n")
    for row in matrix:
        stdout.write("║ ·")
        for col in row:
            cell = []
            cell.append("A") if col[0] == 1 else cell.append(" ")
            cell.append("B") if col[1] == 1 else cell.append(" ")
            cell.append(symbols[0]) if col[2] == 1 else cell.append(" ")
            if game != "escalation":
                cell.append(symbols[1]) if col[3] == 1 else cell.append(" ")
            else:
                cell.append(" ")
            stdout.write("".join(cell) + "·")
        stdout.write(" ║")
        stdout.write("\n")
    stdout.write("╚═" + "═════" * grid_size[0] + "══╝\n\r")
    stdout.flush()


def overlaps_entity(a, b):
    """
    :param a: (X, Y) tuple for entity 1
    :param b: (X, Y) tuple for entity 2
    :return: True if they are on the same cell, False otherwise
    """
    return (a == b).all()


def place_entity_in_unoccupied_cell(used_coordinates, grid_dims):
    """
    Returns a random unused coordinate.
    :param used_coordinates: a list of already used coordinates
    :param grid_dims: dimensions of the grid so we know what a valid coordinate is
    :return: the chosen x, y coordinate
    """
    all_coords = list(product(list(range(grid_dims[0])), list(range(grid_dims[1]))))

    for coord in used_coordinates:
        for test in all_coords:
            if all(test == coord):
                all_coords.remove(test)

    return choice(all_coords)


def spawn_plants(grid_dims, how_many, used_coordinates):
    new_plants = []
    for x in range(how_many):
        new_plant = zeros(2, dtype=uint8)
        new_pos = place_entity_in_unoccupied_cell(
            grid_dims=grid_dims, used_coordinates=new_plants + used_coordinates
        )
        new_plant[0], new_plant[1] = new_pos
        new_plants.append(new_plant)
    return new_plants


def respawn_plants(plants, tagged_plants, grid_dims, used_coordinates):
    for tagged_plant in tagged_plants:
        new_plant = zeros(2, dtype=uint8)
        new_pos = place_entity_in_unoccupied_cell(
            grid_dims=grid_dims, used_coordinates=plants + used_coordinates
        )
        new_plant[0], new_plant[1] = new_pos
        plants[tagged_plant] = new_plant
    return plants


def does_not_respawn_plants(plants, tagged_plants, grid_dims, used_coordinates):
    for tagged_plant in tagged_plants:
        new_plant = zeros(2, dtype=uint8)
        new_plant[0], new_plant[1] = 255, 255
        plants[tagged_plant] = new_plant
    return plants


class AbstractMarkovStagHuntEnv(Env, ABC):
    metadata = {"render.modes": ["human", "array"], "obs.types": ["image", "coords"]}

    def __init__(self, grid_size=(5, 5), obs_type="image", enable_multiagent=False):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """

        total_cells = grid_size[0] * grid_size[1]
        if total_cells < 3:
            raise AttributeError(
                "Grid is too small. Please specify a larger grid size."
            )
        if obs_type not in self.metadata["obs.types"]:
            raise AttributeError(
                'Invalid observation type provided. Please specify "image" or "coords"'
            )
        if grid_size[0] >= 255 or grid_size[1] >= 255:
            raise AttributeError(
                "Grid is too large. Please specify a smaller grid size."
            )

        super(AbstractMarkovStagHuntEnv, self).__init__()

        self.obs_type = obs_type
        self.done = False
        self.enable_multiagent = enable_multiagent

    def step(self, actions):
        """
        Run one timestep of the environment's dynamics.
        :param actions: ints signifying actions for the agents. You can pass one, in which case the second agent does a
                        random move, or two, in which case each agent takes the specified action.
        :return: observation, rewards, is the game done, additional info
        """
        return self.game.update(actions)

    def reset(self):
        """
        Reset the game state
        :return: initial observation
        """
        self.game.reset_entities()
        self.done = False
        return self.game.get_observation()

    def render(self, mode="human", obs=None):
        """
        :param obs: observation data (passed for coord observations so we dont have to run the function twice)
        :param mode: rendering mode
        :return:
        """
        if mode == "human":
            if self.obs_type == "image":
                self.game.RENDERER.render_on_display()
            else:
                if self.game.RENDERER:
                    self.game.RENDERER.update()
                    self.game.RENDERER.render_on_display()
                else:
                    if obs is not None:
                        print_matrix(obs, self.game_title, self.game.GRID_DIMENSIONS)
                    else:
                        print_matrix(
                            self.game.get_observation(),
                            self.game_title,
                            self.game.GRID_DIMENSIONS,
                        )
        elif mode == "array":
            print_matrix(
                self.game._coord_observation(),
                self.game_title,
                self.game.GRID_DIMENSIONS,
            )

    def close(self):
        """
        Closes all needed resources
        :return:
        """
        if self.game.RENDERER:
            self.game.RENDERER.quit()


class AbstractGridGame(ABC):
    def __init__(self, grid_size, screen_size, obs_type, enable_multiagent):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        :param enable_multiagent: Boolean signifying if the env will be used to train multiple agents or one.
        """
        if screen_size[0] * screen_size[1] == 0:
            raise AttributeError(
                "Screen size is too small. Please provide larger screen size."
            )

        # Config
        self._renderer = None  # placeholder renderer
        self._obs_type = obs_type  # record type of observation as attribute
        self._grid_size = grid_size  # record grid dimensions as attribute
        self._enable_multiagent = enable_multiagent

        self._a_pos = zeros(
            2, dtype=uint8
        )  # create empty coordinate tuples for the agents
        self._b_pos = zeros(2, dtype=uint8)

    """
    Observations
    """

    def get_observation(self):
        """
        :return: observation of the current game state
        """
        return (
            self.RENDERER.update()
            if self._obs_type == "image"
            else self._coord_observation()
        )

    def _coord_observation(self):
        return array(self.AGENTS)

    def _flip_coord_observation_perspective(self, a_obs):
        """
        Transforms the default observation (which is "from the perspective of agent A" as it's coordinates are in the
        first index) into the "perspective of agent B" (by flipping the positions of the A and B coordinates in the
        observation array)
        :param a_obs: Original observation
        :return: Original observation, from the perspective of agent B
        """
        ax, ay = a_obs[0], a_obs[1]
        bx, by = a_obs[2], a_obs[3]

        b_obs = a_obs.copy()
        b_obs[0], b_obs[1] = bx, by
        b_obs[2], b_obs[3] = ax, ay
        return b_obs

    """
    Movement Methods
    """

    def _move_dispatcher(self):
        """
        Helper function for streamlining entity movement.
        """
        return {
            LEFT: self._move_left,
            DOWN: self._move_down,
            RIGHT: self._move_right,
            UP: self._move_up,
            STAND: self._stand,
        }

    def _move_entity(self, entity_pos, action):
        """
        Move the specified entity
        :param entity_pos: starting position
        :param action: which direction to move
        :return: new position tuple
        """
        return self._move_dispatcher()[action](entity_pos)

    def _move_agents(self, agent_moves):
        self.A_AGENT = self._move_entity(self.A_AGENT, agent_moves[0])
        self.B_AGENT = self._move_entity(self.B_AGENT, agent_moves[1])

    def _reset_agents(self):
        """
        Place agents in the top left and top right corners.
        :return:
        """
        self.A_AGENT, self.B_AGENT = [0, 0], [self.GRID_W - 1, 0]

    def _random_move(self, pos):
        """
        :return: a random direction
        """
        options = [LEFT, RIGHT, UP, DOWN]
        if pos[0] == 0:
            options.remove(LEFT)
        elif pos[0] == self.GRID_W - 1:
            options.remove(RIGHT)

        if pos[1] == 0:
            options.remove(UP)
        elif pos[1] == self.GRID_H - 1:
            options.remove(DOWN)

        return choice(options)

    def _seek_entity(self, seeker, target):
        """
        Returns a move which will move the seeker towards the target.
        :param seeker: entity doing the following
        :param target: entity getting followed
        :return: up, left, down or up move
        """
        seeker = seeker.astype(int)
        target = target.astype(int)
        options = []

        if seeker[0] < target[0]:
            options.append(RIGHT)
        if seeker[0] > target[0]:
            options.append(LEFT)
        if seeker[1] > target[1]:
            options.append(UP)
        if seeker[1] < target[1]:
            options.append(DOWN)

        if not options:
            options = [STAND]
        shipback = choice(options)

        return shipback

    def _move_left(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_x = pos[0] - 1
        if new_x == -1:
            new_x = 0
        return new_x, pos[1]

    def _move_right(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_x = pos[0] + 1
        if new_x == self.GRID_W:
            new_x = self.GRID_W - 1
        return new_x, pos[1]

    def _move_up(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_y = pos[1] - 1
        if new_y == -1:
            new_y = 0
        return pos[0], new_y

    def _move_down(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_y = pos[1] + 1
        if new_y == self.GRID_H:
            new_y = self.GRID_H - 1
        return pos[0], new_y

    def _stand(self, pos):
        return pos

    """
    Properties
    """

    @property
    def GRID_DIMENSIONS(self):
        return self.GRID_W, self.GRID_H

    @property
    def GRID_W(self):
        return int(self._grid_size[0])

    @property
    def GRID_H(self):
        return int(self._grid_size[1])

    @property
    def AGENTS(self):
        return [self._a_pos, self._b_pos]

    @property
    def A_AGENT(self):
        return self._a_pos

    @A_AGENT.setter
    def A_AGENT(self, new_pos):
        self._a_pos[0], self._a_pos[1] = new_pos[0], new_pos[1]

    @property
    def B_AGENT(self):
        return self._b_pos

    @B_AGENT.setter
    def B_AGENT(self, new_pos):
        self._b_pos[0], self._b_pos[1] = new_pos[0], new_pos[1]

    @property
    def RENDERER(self):
        return self._renderer

    @property
    def COORD_OBS(self):
        return self._coord_observation()


class StagHunt(AbstractGridGame):
    def __init__(
            self,
            stag_reward,
            stag_follows,
            run_away_after_maul,
            opponent_policy,
            forage_quantity,
            forage_reward,
            mauling_punishment,
            # Super Class Params
            window_title,
            grid_size,
            screen_size,
            obs_type,
            load_renderer,
            enable_multiagent,
            will_respawn_plants=True,
            will_respawn_stag=True,
            move_closer_reward=False,
    ):
        """
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param stag_follows: Should the stag seek out the nearest agent (true) or take a random move (false)
        :param run_away_after_maul: Does the stag stay on the same cell after mauling an agent (true) or respawn (false)
        :param forage_quantity: How many plants will be placed on the board.
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """

        super(StagHunt, self).__init__(
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
        )

        self.will_respawn_plants = will_respawn_plants
        self.will_respawn_stag = will_respawn_stag
        self.move_closer_reward = move_closer_reward

        # Config
        self._stag_follows = stag_follows
        self._run_away_after_maul = run_away_after_maul
        self._opponent_policy = opponent_policy

        # Reinforcement Variables
        self._stag_reward = stag_reward  # record RL values as attributes
        self._forage_quantity = forage_quantity
        self._forage_reward = forage_reward
        self._mauling_punishment = mauling_punishment

        # State Variables
        self._tagged_plants = []  # harvested plants that need to be re-spawned

        # Entity Positions
        self._stag_pos = zeros(2, dtype=uint8)
        self._plants_pos = []
        self.reset_entities()  # place the entities on the grid

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == "image" or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.hunt_renderer import HuntRenderer

            self._renderer = HuntRenderer(
                game=self, window_title=window_title, screen_size=screen_size
            )

    """
    Collision Logic
    """

    def _overlaps_plants(self, a, plants):
        """
        :param a: (X, Y) tuple for entity 1
        :param plants: Array of (X, Y) tuples corresponding to plant positions
        :return: True if a overlaps any of the plants, False otherwise
        """
        for x in range(0, len(plants)):
            pos = plants[x]
            if a[0] == pos[0] and a[1] == pos[1]:
                self._tagged_plants.append(x)
                return True
        return False

    """
    State Updating Methods
    """

    def _calc_reward(self):
        """
        Calculates the reinforcement rewards for the two agents.
        :return: A tuple R where R[0] is the reinforcement for A_Agent, and R[1] is the reinforcement for B_Agent
        """

        if overlaps_entity(self.A_AGENT, self.STAG):
            if overlaps_entity(self.B_AGENT, self.STAG):
                rewards = self._stag_reward, self._stag_reward  # Successful stag hunt
            else:
                if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                    rewards = (
                        self._mauling_punishment,
                        self._forage_reward,
                    )  # A is mauled, B foraged
                else:
                    rewards = (
                        self._mauling_punishment,
                        0,
                    )  # A is mauled, B did not forage

        elif overlaps_entity(self.B_AGENT, self.STAG):
            """
            we already covered the case where a and b are both on the stag,
            so we can skip that check here
            """
            if self._overlaps_plants(self.A_AGENT, self.PLANTS):
                rewards = (
                    self._forage_reward,
                    self._mauling_punishment,
                )  # A foraged, B is mauled
            else:
                rewards = 0, self._mauling_punishment  # A did not forage, B is mauled

        elif self._overlaps_plants(self.A_AGENT, self.PLANTS):
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                rewards = (
                    self._forage_reward,
                    self._forage_reward,
                )  # Both agents foraged
            else:
                rewards = self._forage_reward, 0  # Only A foraged

        else:
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                rewards = 0, self._forage_reward  # Only B foraged
            else:
                rewards = 0, 0  # No one got anything
        return float(rewards[0]), float(rewards[1])

    def update(self, agent_moves):
        """
        Takes in agent actions and calculates next game state.
        :param agent_moves: If multi-agent, a tuple of actions. Otherwise a single action and the opponent takes an
                            action according to its established policy.
        :return: observation, rewards, is the game done
        """
        # Move Entities
        self._move_stag()
        if self._enable_multiagent:
            self._move_agents(agent_moves=agent_moves)
        else:
            if self._opponent_policy == "random":
                self._move_agents(
                    agent_moves=[agent_moves, self._random_move(self.B_AGENT)]
                )
            elif self._opponent_policy == "pursuit":
                self._move_agents(
                    agent_moves=[
                        agent_moves,
                        self._seek_entity(self.B_AGENT, self.STAG),
                    ]
                )

        # Get Rewards
        iteration_rewards = self._calc_reward()

        # Reset prey if it was caught
        if iteration_rewards == (self._stag_reward, self._stag_reward):
            if self.will_respawn_stag:
                self.STAG = place_entity_in_unoccupied_cell(
                    grid_dims=self.GRID_DIMENSIONS,
                    used_coordinates=self.PLANTS + self.AGENTS + [self.STAG],
                )
            else:
                self.STAG = [255, 255]
        elif (
                self._run_away_after_maul and self._mauling_punishment in iteration_rewards
        ):
            self.STAG = place_entity_in_unoccupied_cell(
                grid_dims=self.GRID_DIMENSIONS,
                used_coordinates=self.PLANTS + self.AGENTS + [self.STAG],
            )
        elif self._forage_reward in iteration_rewards:
            if self.will_respawn_plants:
                new_plants = respawn_plants(
                    plants=self.PLANTS,
                    tagged_plants=self._tagged_plants,
                    grid_dims=self.GRID_DIMENSIONS,
                    used_coordinates=self.AGENTS + [self.STAG],
                )
                self._tagged_plants = []
                self.PLANTS = new_plants
            else:
                new_plants = does_not_respawn_plants(
                    plants=self.PLANTS,
                    tagged_plants=self._tagged_plants,
                    grid_dims=self.GRID_DIMENSIONS,
                    used_coordinates=self.AGENTS + [self.STAG],
                )
                self._tagged_plants = []
                self.PLANTS = new_plants

        # todo temp
        is_done = False
        if iteration_rewards[1] >= 2:
            is_done = True

        iteration_rewards = self.add_additional_rewards(iteration_rewards)

        obs = self.get_observation()
        info = {}

        if self._enable_multiagent:
            if self._obs_type == "coords":
                return (
                    (obs, self._flip_coord_observation_perspective(obs)),
                    iteration_rewards,
                    is_done,
                    info,
                )
            else:
                return (obs, obs), iteration_rewards, is_done, info
        else:
            return obs, iteration_rewards[0], is_done, info

    def _coord_observation(self):
        """
        :return: list of all the entity coordinates
        """
        shipback = [self.A_AGENT, self.B_AGENT, self.STAG]
        shipback = shipback + self.PLANTS
        return array(shipback).flatten()

    """
    Movement Methods
    """

    def _seek_agent(self, agent_to_seek):
        """
        Moves the stag towards the specified agent
        :param agent_to_seek: agent to pursue
        :return: new position tuple for the stag
        """
        agent = self.A_AGENT
        if agent_to_seek == "b":
            agent = self.B_AGENT

        move = self._seek_entity(self.STAG, agent)

        return self._move_entity(self.STAG, move)

    def _move_stag(self):
        """
        Moves the stag towards the nearest agent.
        :return:
        """
        if self._stag_follows:
            stag, agents = self.STAG, self.AGENTS
            a_dist = hypot(
                int(agents[0][0]) - int(stag[0]), int(agents[0][1]) - int(stag[1])
            )
            b_dist = hypot(
                int(agents[1][0]) - int(stag[0]), int(agents[1][1]) - int(stag[1])
            )

            if a_dist < b_dist:
                agent_to_seek = "a"
            else:
                agent_to_seek = "b"

            self.STAG = self._seek_agent(agent_to_seek)
        else:
            self.STAG = self._move_entity(self.STAG, self._random_move(self.STAG))

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self.STAG = [self.GRID_W // 2, self.GRID_H // 2]
        self.PLANTS = spawn_plants(
            grid_dims=self.GRID_DIMENSIONS,
            how_many=self._forage_quantity,
            used_coordinates=self.AGENTS + [self.STAG],
        )

    """
    Properties
    """

    @property
    def STAG(self):
        return self._stag_pos

    @STAG.setter
    def STAG(self, new_pos):
        self._stag_pos[0], self._stag_pos[1] = new_pos[0], new_pos[1]

    @property
    def PLANTS(self):
        return self._plants_pos

    @PLANTS.setter
    def PLANTS(self, new_pos):
        self._plants_pos = new_pos

    @property
    def ENTITY_POSITIONS(self):
        return {
            "a_agent": self.A_AGENT,
            "b_agent": self.B_AGENT,
            "stag": self.STAG,
            "plants": self.PLANTS,
        }

    def add_additional_rewards(self, iteration_rewards):
        rewards = iteration_rewards
        if self.move_closer_reward:
            # a reward that is proportional to the distance between the agents and the stag
            # like a force field that pulls the agents towards the stag
            MAX_REWARD_STAG = 0.1
            a_dist = calculate_distance(self.A_AGENT, self.STAG)
            b_dist = calculate_distance(self.B_AGENT, self.STAG)
            if a_dist == 0:
                a_dist = 1
            if b_dist == 0:
                b_dist = 1
            rewards = (rewards[0] + MAX_REWARD_STAG / a_dist, rewards[1] + MAX_REWARD_STAG / b_dist)
            MAX_REWARD_PLANT = 0.05
            a_dist_to_closest_plant = min([calculate_distance(self.A_AGENT, plant) for plant in self.PLANTS])
            b_dist_to_closest_plant = min([calculate_distance(self.B_AGENT, plant) for plant in self.PLANTS])
            if a_dist_to_closest_plant == 0:
                a_dist_to_closest_plant = 1
            if b_dist_to_closest_plant == 0:
                b_dist_to_closest_plant = 1
            rewards = (rewards[0] + MAX_REWARD_PLANT / a_dist_to_closest_plant,
                       rewards[1] + MAX_REWARD_PLANT / b_dist_to_closest_plant)
        return rewards


class HuntEnv(AbstractMarkovStagHuntEnv):
    def __init__(
            self,
            grid_size=(5, 5),
            screen_size=(600, 600),
            obs_type="image",
            enable_multiagent=False,
            opponent_policy="random",
            load_renderer=False,
            stag_follows=True,
            run_away_after_maul=False,
            forage_quantity=2,
            stag_reward=5,
            forage_reward=1,
            mauling_punishment=-5,
            respawn_plants=True,
            respawn_stag=True,
            move_closer_reward=False,
    ):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        :param stag_follows: Should the stag seek out the nearest agent (true) or take a random move (false)
        :param run_away_after_maul: Does the stag stay on the same cell after mauling an agent (true) or respawn (false)
        :param forage_quantity: How many plants will be placed on the board.
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """
        if not (stag_reward > forage_reward >= 0 > mauling_punishment):
            raise AttributeError(
                "The game does not qualify as a Stag Hunt, please change parameters so that "
                "stag_reward > forage_reward >= 0 > mauling_punishment"
            )
        if mauling_punishment == forage_reward:
            raise AttributeError(
                "Mauling punishment and forage reward are equal."
                " Game logic will not function properly."
            )
        total_cells = grid_size[0] * grid_size[1]
        if (
                forage_quantity >= total_cells - 3
        ):  # -3 is for the cells occupied by the agents and stag
            raise AttributeError(
                "Forage quantity is too high. The plants will not fit on the grid."
            )
        if total_cells < 3:
            raise AttributeError(
                "Grid is too small. Please specify a larger grid size."
            )

        super(HuntEnv, self).__init__(
            grid_size=grid_size, obs_type=obs_type, enable_multiagent=enable_multiagent
        )

        self.game_title = "hunt"
        self.stag_reward = stag_reward
        self.forage_reward = forage_reward
        self.mauling_punishment = mauling_punishment
        self.reward_range = (mauling_punishment, stag_reward)

        window_title = (
                "OpenAI Gym - Stag Hunt (%d x %d)" % grid_size
        )  # create game representation
        self.game = StagHunt(
            window_title=window_title,
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
            load_renderer=load_renderer,
            stag_reward=stag_reward,
            stag_follows=stag_follows,
            run_away_after_maul=run_away_after_maul,
            forage_quantity=forage_quantity,
            forage_reward=forage_reward,
            mauling_punishment=mauling_punishment,
            opponent_policy=opponent_policy,
            will_respawn_plants=respawn_plants,
            will_respawn_stag=respawn_stag,
            move_closer_reward=move_closer_reward,
        )

        self.action_space = Discrete(5)  # up, down, left, right or stand

        if obs_type == "image":
            self.observation_space = Box(
                0,
                255,
                shape=(grid_size[0] * TILE_SIZE, grid_size[1] * TILE_SIZE, 3),
                dtype=uint8,
            )
        elif obs_type == "coords":
            self.observation_space = Box(
                0, max(grid_size), shape=(6 + forage_quantity * 2,), dtype=uint8
            )


class ZooHuntEnvironment(PettingZooEnv):
    metadata = {"render_modes": ["human", "array"], "name": "hunt_pz"}

    def __init__(
            self,
            grid_size=(5, 5),
            screen_size=(600, 600),
            obs_type="image",
            enable_multiagent=False,
            opponent_policy="random",
            load_renderer=False,
            stag_follows=True,
            run_away_after_maul=False,
            forage_quantity=2,
            stag_reward=5,
            forage_reward=1,
            mauling_punishment=-5,
            respawn_plants=True,
            respawn_stag=True,
            move_closer_reward=False,
    ):
        hunt_env = HuntEnv(
            grid_size,
            screen_size,
            obs_type,
            enable_multiagent,
            opponent_policy,
            load_renderer,
            stag_follows,
            run_away_after_maul,
            forage_quantity,
            stag_reward,
            forage_reward,
            mauling_punishment,
            respawn_plants,
            respawn_stag,
            move_closer_reward,
        )
        super().__init__(og_env=hunt_env)


def default_wrappers(env_init):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


if __name__ == "__main__":
    main()
