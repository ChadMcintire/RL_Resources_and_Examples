import os
import gym
import torch
import numpy as np
import os

from utils import ReplayBuffer, eval_policy
import TD3

def training_loop(args):
   
    #Create gym env
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up directory pathing for saving results and models
    file_name = f"{args.policy}_{args.env}_{args.seed}"

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    #create gym and policy required variables 

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action, 
            "discount": args.discount,
            "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy is scaled with respect to the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    #Load pretrain policy if wished
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    #Create fifo replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0 
    episode_timesteps = 0
    episode_num = 0

    #Training loop starts here
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        # If start time steps are low we are generating data for the 
        # replay buffer
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                     ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        #store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        # Set up for next state
        state = next_state
        episode_reward += reward

        # Train agent collecting sufficient data
        # If the buffer is sufficiently full start the training loop
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #print variables every episode
            #print(f"Total T: {t+1} Episode num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            #Reset environment
            state, done = env.reset(), False
            episode_reward += reward
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: 
                print("saving model")
                policy.save(f"./models/{file_name}")

