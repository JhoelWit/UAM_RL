# UAM_RL
Designing a reinforcement learning air traffic controller agent for use in an Unreal Engine simulator with an Airsim client.

This is a refactored (and still in the refactoring process) version of the [GLUAM](https://github.com/JhoelWit/GLUAM) repository, used for ICRA 2022 and SciTech 2023. 

The goal was to create an air traffic controller agent which could assume the responsibilities of a real life ATC, utilizing urban air mobility to command electrical vertical takeoff and landing vehicles around a vertiport. The problem is formulated as a markov decision process, and PPO is used for the reinforcement learning algorithm. 

Unreal Engine and Airsim is used to create a simple [environment](https://drive.google.com/file/d/1XT5h9mm4l6FRyraBGow9nLI4yc4STpl0/view?usp=sharing) for training and testing, consisting of three helipad objects (2 normal ports and one battery charging port):

![](img/sim_env.gif)

The action space consists of 11 actions, and the state space is formulated as two homogeneous graphs, for the vertiport and eVTOLs respectively. Here is an example of the agent taking three separate actions:

![](img/EVTOL_actions.gif)

The PPO policy uses graph neural networks to extract key features from the two graphs, and propagates them through a mulit-layered perceptron to get log probabilities for the next discrete action:

![](img/GRL_Policy.png)

This codestack was used to train a successful ATC agent which outperformed a common baseline for flight planning (first come first serve) as well as a random agent and one using a denser policy network without graphs:

![](img/case1_plot_final_glass-min.png)

The papers will be attached to this readme once they're published and/or added to arXiv. 

UPDATE: Here is one of the [papers](https://arxiv.org/abs/2302.05849)
