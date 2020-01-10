import os
from collections import deque
import numpy as np
import vizdoom

from paiutils import image
from paiutils.reinforcement import Environment


class Doom(Environment):
    games = ['basic', 'deadly_corridor', 'deathmatch', 'defend_the_center',
             'defend_the_line', 'health_gathering', 'health_gathering_supreme',
             'my_way_home', 'predict_position', 'rocket_basic', 'simpler_basic',
             'take_cover']

    def __init__(self, game, state_shape, preprocess_frame_func,
                 num_stacked_states=3):
        if game not in Doom.games:
            raise ValueError(f'{game} is not a valid doom game.')
        self.game = vizdoom.DoomGame()
        self.game.load_config(
            os.path.join(vizdoom.scenarios_path, f'{game}.cfg')
        )
        self.game.set_doom_scenario_path(
            os.path.join(vizdoom.scenarios_path, f'{game}.wad')
        )
        self.game.init()
        self.num_stacked_states = num_stacked_states
        self.states = deque(maxlen=num_stacked_states)
        self.preprocess_frame_func = preprocess_frame_func
        self.single_state_shape = state_shape
        super().__init__((*state_shape[:-1],
                          state_shape[-1]*num_stacked_states),
                         (self.game.get_available_buttons_size(),))
        self.actions = np.identity(self.action_shape[0]).tolist()

    def _get_state(self):
        state = self.game.get_state()
        if state is not None:
            state = state.screen_buffer
            if state.ndim == 3:
                state = np.swapaxes(state, 2, 0)
                state = np.rot90(state, -1)
                state = image.hflip(state)
                state = image.rgb2bgr(state)
            state = self.preprocess_frame_func(state)
            self.state = state
            self.states.append(state)
        if self.num_stacked_states == 1:
            return self.state
        else:
            if self.state.ndim == 3:
                return np.dstack(self.states)
            elif self.state.ndim < 3:
                return np.hstack(self.states)
            else:
                raise Exception('Too many state dims')

    def reset(self):
        self.state = np.zeros(self.single_state_shape)
        for _ in range(self.num_stacked_states):
            self.states.append(self.state)
        self.game.new_episode()
        return self._get_state()

    def step(self, action):
        action = self.actions[action]
        reward = self.game.make_action(action)
        terminal = self.game.is_episode_finished()
        return self._get_state(), reward, terminal

    def close(self):
        self.game.close()
