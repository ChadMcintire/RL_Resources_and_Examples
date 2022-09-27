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
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) # Step 4.ii pseudocode

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # but clip the noise to keep the action close to original value
            noise = ( 
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip) # Step 4.iii pseudocode


            #paper doesn't say to do this but clamping to the max possible action
            #prevents noise from going over the max
            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action) # Step 4.iii pseudocode

            #Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action) # Step 4.iv pseudocode
            target_Q = torch.min(target_Q1, target_Q2) # Step 4.iv pseudocode
            target_Q = reward + not_done + self.discount * target_Q  # Step 4.iv pseudocode


        #Get current Q estimate
        current_Q1, current_Q2 = self.critic(state, action) # Step 4.v pseudocode

        #Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) # Step 4.v pseudocode

        #Optimize the critic
        self.critic_optimizer.zero_grad() # Step 4.v pseudocode
        critic_loss.backward() # Step 4.v pseudocode
        self.critic_optimizer.step() # Step 4.v pseudocode

        #Delayed policy updates
        if self.total_it % self.policy_freq == 0: # Step 4.vi

            #Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() # Step 4.vi.a

            # Optimize the actor
            self.actor_optimizer.zero_grad() # Step 4.vi.a
            actor_loss.backward() # Step 4.vi.a
            self.actor_optimizer.step() # Step 4.vi.a


            #Update the frozen target models
            soft_update(self.critic_target, self.critic, self.tau) # Step 4.vi.b of pseudocode the first one  
            soft_update(self.actor_target, self.actor, self.tau) # Step 4.vi.b of pseudocode the second one


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
