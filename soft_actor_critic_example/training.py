import argparse
import gym
import torch
from sac import SAC
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from replay_memory import ReplayMemory
import itertools
from utils import validation_episodes

def training_loop(args):
    #set up environment
    env = gym.make(args.env_name)
    val_steps = args.steps_between_validation
    current_reward = 0
    max_reward = 0

    #set up seeding
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #initialize the sac agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    #this is a logger, which is a great idea but might need to be removed
    #so it doesnt' obscure the code for readers
    writer = SummaryWriter(args.logs_dir + "/{}_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                           args.policy, "autotune" if args.automatic_entropy_tuning else "", args.alpha))


    #Memory replay, this helps with non-stationarity and the to help
    #the model not be sequential data for the for the Markov assumption
    memory = ReplayMemory(args.replay_size, args.seed)

    #Training Loop
    total_numsteps = 0
    updates = 0

    for i_episodes in itertools.count(1):
        episode_reward = 0
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample() #Sample random action
            else:
                action = agent.select_action(state) #Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                    writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                    writer.add_scalar("loss/policy", policy_loss, updates)
                    writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                    writer.add_scalar("entropy_temperature/alpha", alpha, updates)
                    updates += 1

            #set the state to the next state and update the variables that depend on the step number
            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward


            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            #update current trajectories by adding them to memory
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            #update the next state to the current state so we can move to the next step
            state = next_state


        #end if max steps reached
        if total_numsteps > args.num_steps:
            break

        writer.add_scalar("reward/train", episode_reward, i_episodes)
        #print every iteration
        #print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episodes, total_numsteps, episode_steps, round(episode_reward, 2)))

        #Validate the learning every several hundred episodes
        if total_numsteps > int(val_steps) and args.eval is True:
            val_steps = int(val_steps) + int(args.steps_between_validation)
            current_reward = validation_episodes(env, total_numsteps, agent, writer, args.render)
            writer.add_scalar("reward/train", current_reward, total_numsteps)

            if max_reward <= current_reward:
                agent.save_checkpoint(args.env_name, suffix=total_numsteps, ckpt_path=args.check_pt_name, alpha=args.alpha)
                max_reward = current_reward

    env.close()
    writer.flush()
    writer.close()

    print("successfully exited")

