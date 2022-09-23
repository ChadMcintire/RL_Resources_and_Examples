import gym
import torch
import numpy as np

from model import Actor, Critic

import TD3

def run(args):
    print("beginning runs")
    #Create gym env
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   

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

    #Load pretrain policy
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    policy.load(f"./models/{file_name}" , evaluate=False)

    #agent.load_checkpoint(args.check_pt_name, evaluate=False)
    while True:
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            env.render()
            #time.sleep(.1)
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        print(episode_reward)
