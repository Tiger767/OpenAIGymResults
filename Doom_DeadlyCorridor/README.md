# Doom: Deadly Corridor

Trial 1
- Episodes of Random Agent (exploring): 0
- Episodes of A2C Agent (convergence episodes): 1500
- Algorithm: Advantage Actor Critic (A2C)
- Actor
  - Model: Input -> Conv(32) -> Conv(64) -> Conv(128) -> Conv(256) -> Conv(512) -> Flatten() -> Dense(3) -> Softmax
    - Conv(filter) = Conv2D(filter, kernel_size=3, strides=2) -> ReLu -> BatchNorm 
  - Optimizer: Adam(.001)
- Critic
  - Model: Input -> Conv(32) -> Conv(64) -> Conv(128) -> Conv(256) -> Conv(512) -> Flatten() -> Dense(1)
    - Conv(filter) = Conv2D(filter, kernel_size=3, strides=2) -> ReLu -> BatchNorm 
  - Optimizer: Adam(.001)
  - Loss: mean squared error
- Agent Parameters
  - Discount Rate (gamma): .99
  - Lambda Rate (enables GAE > 0): 0
  - Memory: 100000
- Learning Parameters
  - Batch Size: 64
  - Mini-Batch size (Sample size from all experience): 10000
  - Epochs (Number of complete gradient steps per episode): 1
- Environment Parameters:
  - Number of Stacked Frames for State: 3
  - Frame Preprocess:
    - Frame -> Grayscale Frame -> Resized Frame 32x64 -> Shrink Top and Bottom by 16px -> Normalize Frame -> (32, 32, 1)
- Results
  - Average Total Reward over 100 Episodes: 537.28
  - Highest Score within best 100-episode: 776.57
  - Lowest Score within best 100-episode: -3.04

Gameplay from the trial 1 agent
![](./a2c_537.28_ep1500.gif)

Note: The weights provided can be loaded with TensorFlow Keras, and the code below uses my [PAI-Utils package](https://pypi.org/project/paiutils/)

Trial 1 Code
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from paiutils.reinforcement import (
    RingMemory
)
from paiutils.reinforcement_agents import (
    A2CAgent
)
from paiutils.neural_network import (
    conv2d
)
from paiutils import image
from doom_env import Doom


def create_amodel_conv(state_shape, action_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = conv2d(32, 3, 2)(inputs)
    x = conv2d(64, 3, 2)(x)
    x = conv2d(128, 3, 2)(x)
    x = conv2d(256, 3, 2)(x)
    x = conv2d(512, 3, 2)(x)
    x = keras.layers.Flatten()(x)
    outputs = dense(action_shape[0], activation='softmax',
                    batch_norm=False)(x)
    
    amodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    amodel.compile(optimizer=keras.optimizers.Adam(.001),
                   loss='mse', experimental_run_tf_function=False)
    amodel.summary()
    return amodel


def create_cmodel_conv(state_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = conv2d(32, 3, 2)(inputs)
    x = conv2d(64, 3, 2)(x)
    x = conv2d(128, 3, 2)(x)
    x = conv2d(256, 3, 2)(x)
    x = conv2d(512, 3, 2)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1)(x)

    cmodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    cmodel.compile(optimizer=keras.optimizers.Adam(.001),
                   loss='mse', experimental_run_tf_function=False)
    cmodel.summary()
    return cmodel


if __name__ == '__main__':
    save_dir = ''
    max_steps = 10000
    windows = image.Windows()
    # preprocess_window = windows.add('Preprocess')
    # windows.start()

    def preprocess(frame):
        frame = image.gray(frame)
        frame = image.resize(frame, (32, 64))
        frame = image.shrink_sides(frame, ts=16, bs=16)
        # windows.set(preprocess_window, frame)
        frame = image.normalize(frame)
        return np.expand_dims(frame, axis=-1)

    env = Doom('deadly_corridor', (32, 32, 1), preprocess,
               num_stacked_states=3)
    
    amodel = create_amodel_conv(env.state_shape, env.action_shape)
    cmodel = create_cmodel_conv(env.state_shape)
    agent = A2CAgent(amodel, cmodel, .99, 
                     create_memory=lambda: RingMemory(100000))

    agent.set_playing_data(training=True, memorizing=True,
                           batch_size=64, mini_batch=10000, epochs=1,
                           entropy_coef=0,
                           verbose=True)
    for ndx in range(50):
        print(f'Save Loop: {ndx}')
        result = env.play_episodes(agent, 20, max_steps,
                                   verbose=True, episode_verbose=False,
                                   render=False)
        agent.save(save_dir, note=f'A2C_{ndx}_{result}')

    agent.set_playing_data(training=False, memorizing=False, verbose=True)
    avg = env.play_episodes(agent, 100, max_steps,
                            verbose=True, episode_verbose=False,
                            render=False)
    print(len(agent.states))
    print(avg)
```

Doom Environment Code from doom_env.py
```python
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
```
