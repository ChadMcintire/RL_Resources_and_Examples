import copy
import numpy as np
from utils import soft_update

import torch

from model import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(
            self, 
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

            # Set up Actor, actor target, and optimizer
            self.actor = Actor(state_dim, action_dim, max_action).to(device) #Step 1 pseudocode
            self.actor_target = copy.deepcopy(self.actor) #Step 2 Pseudocode
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

            # Set up Critic, critic target, and optimizer
            self.critic = Critic(state_dim, action_dim).to(device) #Step 1 pseudocode
            self.critic_target = copy.deepcopy(self.critic) #Step 2 Pseudocode
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

            #Set up hyper parameters
            self.max_action = max_action
            self.discount = discount
            self.tau = tau
            self.policy_noise = policy_noise
            self.noise_clip = noise_clip
            self.policy_freq = policy_freq

            self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        #Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)


            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            #Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done + self.discount * target_Q


        #Get current Q estimate
        current_Q1, current_Q2 = self.critic(state, action)

        #Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        #Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            #Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


            #Update the frozen target models
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)


    #Save the actor and critic and their optimizers
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    #Save the actor and critic and their optimizers
    def load(self, filename, evaluate=True):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        if evaluate:
            self.critic.eval()
            self.critic_target.eval()
            self.actor.eval()
            self.actor_target.eval()

        else:
            self.critic.train()
            self.critic_target.train()
            self.actor.train()
            self.actor_target.train()
