# Doom: Basic

Trial 1
- Episodes of Random Agent (exploring): 0
- Episodes of Policy Gradient Agent (convergence episodes): 200
- Algorithm: Policy Gradients (No Critic) using an encoder (autoencoder) for inputs
- Autoencoder
  - Encoder Model: Input((64, 64, 1)) -> Conv(32) -> Conv(64) -> Conv(128) -> Conv(256) -> Conv(256) -> Conv(256)
    - Conv(filter) = Conv2D(filter, kernel_size=3, strides=2) -> ReLu -> BatchNorm 
  - Optimizer: Adam(.001)
  - Loss: mean absolute error
  - Dataset: ~20,000 images of the environment gathered by a random agent
  - Epochs: ~400
  - Batch Size: 32, 256
- Actor
  - Model: Input -> Dense(64) -> ReLu -> BatchNorm -> Dense(64) -> ReLu -> BatchNorm -> Dense(3) -> Softmax
  - Optimizer: Adam(.001)
- Agent Parameters
  - Discount Rate: .99
  - Memory: 100000
- Learning Parameters
  - Batch Size: 32
  - Mini-Batch size (Sample size from all experience): 10000
  - Epochs (Number of complete gradient steps per episode): 1
- Environment Parameters:
  - Number of Stacked Frames for State: 3
  - Frame Preprocess:
    - Frame -> Grayscale Frame -> Resized Frame 64x64 -> Encoded (256,)
- Results
  - Average Total Reward over 100 Episodes: 38.97
  - Highest Score within best 100-episode: 95
  - Lowest Score within best 100-episode: -410

Trial 2
- Episodes of Random Agent (exploring): 0
- Episodes of A2C Agent (convergence episodes): 520
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
  - Average Total Reward over 100 Episodes: 81.22
  - Highest Score within best 100-episode: 95
  - Lowest Score within best 100-episode: 16

Note: The weights provided can be loaded with TensorFlow Keras, and the code below uses my [PAI-Utils package](https://pypi.org/project/paiutils/)

Trial 1 Code
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from paiutils.reinforcement import (
    PGAgent, RingMemory
)
from paiutils.neural_network import (
    dense, conv2d
)
from paiutils.autoencoder import (
    AutoencoderPredictor, AutoencoderTrainner
)
from paiutils.util_funcs import save_h5py
from paiutils import image
from doom_env import Doom

def create_amodel_dense(state_shape, action_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = dense(64)(inputs)
    x = dense(64)(x)
    outputs = dense(action_shape[0], activation='softmax',
                    batch_norm=False)(x)
    
    amodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    amodel.compile(optimizer=keras.optimizers.Adam(.001),
                   loss='mse', experimental_run_tf_function=False)
    amodel.summary()
    return amodel

def create_autoencoder():
    inputs = keras.layers.Input(shape=(64, 64, 1)) # 64
    x = conv2d(32, kernel_size=3, strides=2)(inputs) # 32
    x = conv2d(64, kernel_size=3, strides=2)(x) # 16
    x = conv2d(128, kernel_size=3, strides=2)(x) # 8
    x = conv2d(256, kernel_size=3, strides=2)(x) # 4
    x = conv2d(256, kernel_size=3, strides=2)(x) # 2
    x = conv2d(256, kernel_size=3, strides=2)(x) # 1

    encoder_model = keras.Model(inputs=inputs, outputs=x)

    x = conv2d(256, kernel_size=3, strides=2, transpose=True)(x) # 2
    x = conv2d(256, kernel_size=3, strides=2, transpose=True)(x) # 4
    x = conv2d(256, kernel_size=3, strides=2, transpose=True)(x) # 8
    x = conv2d(128, kernel_size=3, strides=2, transpose=True)(x) # 16
    x = conv2d(64, kernel_size=3, strides=2, transpose=True)(x) # 32
    x = conv2d(32, kernel_size=3, strides=2, transpose=True)(x) # 64
    outputs = conv2d(1, kernel_size=3, activation='tanh',
                     batch_norm=False)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(.001, amsgrad=True),
                  loss='mae')
    model.summary()
    return model, encoder_model

if __name__ == '__main__':
    save_dir = ''
    max_steps = 300
    encoder_path = f'{save_dir}autoencoder'
    encoder_data_path = None # f'{save_dir}autoencoder_data.h5'
    windows = image.Windows()

    if encoder_path is None:
        if encoder_data_path is None:
            def preprocess(frame):
                frame = image.gray(frame)
                frame = image.resize(frame, (64, 64))
                frame = np.expand_dims(frame, axis=-1)
                return frame
            env = Doom('basic', (64, 64, 1),
                       preprocess_frame_func=preprocess,
                       num_stacked_states=1)
            # Use PGAgent to explore with a random agent
            amodel = create_amodel_dense(env.state_shape, env.action_shape)
            agent = PGAgent(amodel, .99,
                            create_memory=lambda: RingMemory(100000))
            agent.set_playing_data(training=False, memorizing=True)
            env.play_episodes(agent, 200, max_steps, random=True,
                              verbose=True, episode_verbose=False,
                              render=True)
            data = {'train_x': image.normalize(agent.states.array())}
            agent.forget()
            env.play_episodes(agent, 10, max_steps, random=True,
                              verbose=True, episode_verbose=False,
                              render=True)
            data['validation_x'] = image.normalize(agent.states.array())
            save_h5py(f'{save_dir}autoencoder_data.h5', data)
            encoder_data_path = f'{save_dir}autoencoder_data.h5'
            env.close()
            del env, agent, data, qmodel
        model, encoder_model = create_autoencoder()
        trainner = AutoencoderTrainner(model, encoder_data_path,
                                       encoder_model=encoder_model)
        trainner.train(400, batch_size=256)
        encoder_path = trainner.save(save_dir)

        # View some Results
        windows.stop()
        windows.start()
        window = windows.add('Autoencoder Results')
        autoencoder = AutoencoderPredictor(encoder_path)
        for x in trainner.validation_data[0]:
            pred = autoencoder.predict(x)
            pred = image.denormalize(pred)
            x = image.denormalize(x)
            windows.set(window, np.hstack([x,pred]))
            if input('Enter to view next image (q to quit)') == 'q':
                break
        windows.stop()
        del autoencoder, trainner, window
    encoder = AutoencoderPredictor(encoder_path, uses_encoder_model=True)
    def preprocess(frame):
        frame = image.gray(frame)
        frame = image.resize(frame, (64, 64))
        frame = np.expand_dims(frame, axis=-1)
        encoded_frame = encoder.predict(image.normalize(frame))
        return encoded_frame.flatten()

    env = Doom('basic', (256,), preprocess,
               num_stacked_states=3)

    policy = None
    amodel = create_amodel_dense(env.state_shape, env.action_shape)
    agent = PGAgent(amodel, .99, create_memory=lambda: RingMemory(100000),
                    policy=policy)

    agent.set_playing_data(training=True, memorizing=True,
                           batch_size=32, mini_batch=10000, epochs=1, 
                           entropy_coef=0,
                           verbose=True)
    for ndx in range(4):
        print(f'Save Loop: {ndx}')
        result = env.play_episodes(agent, 50, max_steps,
                                   verbose=True, episode_verbose=False,
                                   render=False)
        agent.save(save_dir, note=f'PG_{ndx}_{result}')

    agent.set_playing_data(training=False, memorizing=False)
    avg = env.play_episodes(agent, 100, max_steps,
                            verbose=True, episode_verbose=False,
                            render=False)
    env.close()
    print(len(agent.states))
    print(avg)
```

Trial 2 Code
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
    save_dir = 'basic_saves/'
    max_steps = 300
    windows = image.Windows()
    #preprocess_window = windows.add('Preprocess')
    #windows.start()

    def preprocess(frame):
        frame = image.gray(frame)
        frame = image.resize(frame, (32, 64))
        frame = image.shrink_sides(frame, ts=16, bs=16)
        #windows.set(preprocess_window, frame)
        frame = image.normalize(frame)
        return np.expand_dims(frame, axis=-1)

    env = Doom('basic', (32, 32, 1), preprocess,
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

Gameplay from the trial 2 agent
![](./a2c_81.22_ep520.gif)
