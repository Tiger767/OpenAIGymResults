# MountainCar-v0

Trial 1
- Code and weights can be found [here](https://github.com/Tiger767/PAI-Utils/blob/master/examples/reinforcement/reinforcement3.ipynb)
- Episodes of Random Agent (exploring): 0
- Episodes of A2C Agent (convergence episodes): 224
- Algorithm: Advantage Actor Critic (A2C)
- Actor
  - Model: Input -> Dense(64) -> ReLu -> BatchNorm -> Dense(64) -> ReLu -> BatchNorm -> Dense(3) -> Softmax
  - Optimizer: Adam(.001)
- Critic
  - Model: Input -> Dense(64) -> ReLu -> BatchNorm -> Dense(64) -> ReLu -> BatchNorm -> Dense(1)
  - Optimizer: Adam(.001)
  - Loss: mean squared error
- Agent Parameters
  - Discount Rate: .99
  - Lambda Rate: .95
  - Memory: 20000
- Learning Parameters
  - Batch Size: 16
  - Mini-Batch size (Sample size from all experience): 1024
  - Epochs * Repeat (Number of complete gradient steps per episode): 5 (repeat)
- Results
  - Average Total Reward over 100 Episodes: -105.48

Gameplay from the trial 1 agent
[![Gameplay of the trial 1 agent](http://img.youtube.com/vi/V2eJdqS8T9Q/0.jpg)](https://www.youtube.com/watch?v=V2eJdqS8T9Q)
