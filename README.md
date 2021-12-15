# Muscle-Actuated-Arm-DeepRL

This repository was created as part of the Master's course "Data-integrated Simulation Science A, Part B: In silico models of coupled biological systems" at the University of Stuttgart in the winter term 2021/22. 

The main goal of the project was to build a Hill-type muscle model of an arm and then learn the necessary muscle activation to perform a certain movement.
Drawing inspiration from basketball, Deep Reinforcement Learning is used to train an agent to throw a ball into a box.
To be precise, we used Deep Deterministic Policy Gradient (DDPG) as a suitable algorithm for contiuous action and state space.

![Performance of the trained RL agent in a given setting.](../Results/episode_hit.mov)

![Reward over episodes for a simple example. The training was stopped after the first make.](../Results/fast_learner_reward.png)

### Prerequisites

The entire project runs using MATLAB/Simulink and requires the following add-ons to be installed:
- Simscape Multibody
- Reinforcement Learning Toolbox
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional)

### Simulink Model

The enviroment and the learning "loop" are set up in the Simulink model `throwing_arm.slx`. 
The main parts of this are:
- The environment consisting of
    - an upper and lower arm that are connected via a rotatory joint. The upper arm is fixed in space, while the lower arm can be actuated with the monoarticular muslce, with two inputs, one for extension and one for flexion.
    - the box
    - the ball that is connected to the lower arm at first but is released after a certain time
- The RL agent (this block is shipped with the RL Toolbox).
- Subsystems to calculate the reward and check the termination condition.
- A small logicical unit that connects the agents action signal with the desired the muscle input.

### Training & Testing

Training is done in MATLAB where the environment as well as the actor and critic for the DDPG, including the structure of their Deep Neural Networks can be specified.
Further parameters, such as the learning rates, the discount factor, the action noise, the maximum number of time steps per episode, and so on can be edited.
For more detailed information on this, please refer to the [offical Mathworks Documentation](https://de.mathworks.com/products/reinforcement-learning.html).

For demonstration purposes, two agents were stored and are being made available in this repository.
To test their performance, please execute `test_current_agent.m` or `test_current_agent2.m`.

