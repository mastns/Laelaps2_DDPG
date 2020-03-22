"""
Deep Determinitic Policy Gradient (DDPG) Network & Agent based on:
1) Max Lapan's implementation in the book Deep Reinforcement Learning Hands-On: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
2) Pytorch 
3) PTAN (PyTorch AgentNet) package for RL https://github.com/Shmuma/ptan
"""

import ptan
import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_layer1_actor = rospy.get_param("/hidden_layer1_actor")
hidden_layer2_actor = rospy.get_param("/hidden_layer2_actor")
hidden_layer1_critic = rospy.get_param("/hidden_layer1_critic")
hidden_layer2_critic = rospy.get_param("/hidden_layer2_critic")

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_layer1_actor),
            nn.ReLU(),
            nn.Linear(hidden_layer1_actor, hidden_layer2_actor),
            nn.ReLU(),
            nn.Linear(hidden_layer2_actor, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, hidden_layer1_critic),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(hidden_layer1_critic + act_size, hidden_layer2_critic),
            nn.ReLU(),
            nn.Linear(hidden_layer2_critic, 1)  # 1 corresponds to Q value
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))  # cat for catkin

mu = rospy.get_param("/ou_mu")
theta = rospy.get_param("/ou_theta")
sigma = rospy.get_param("/ou_sigma")
epsilon = rospy.get_param("/ou_epsilon")

class AgentDDPG(ptan.agent.BaseAgent):
    """
    Exploration using Orstein-Uhlenbeck exploration process
    """

    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=mu, ou_teta=theta, ou_sigma=sigma, ou_epsilon=epsilon):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)  # outputs deterministic actions
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []  # next agent states
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)  # new agent states
        else:
            new_a_states = agent_states

        # new agent state is saved as it has the noise for the next step
        return actions, new_a_states
