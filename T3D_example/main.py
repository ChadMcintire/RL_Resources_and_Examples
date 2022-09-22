import argparse
import os
import gym
import torch
import numpy as np
import os

import utils
import TD3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("-------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("----------------------------------------------------------")
    return avg_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help="Policy name is (TD3, DDPG or OurDDPG)")
    parser.add_argument("--env", default="HalfCheetah-v2", help="OpenAI gym environment name (default=HalfCheetah-v2)")
    parser.add_argument("--seed", default=0, type=int, help = "Sets Gym, Pytorch and Numpy seeds (default=0)")
    parser.add_argument("--start_timesteps", default=25e3, type=int, help="Time steps initial random policy is used, (default=25,000)")
    parser.add_argument("--eval_freq", default=5e3, type=int, help="How often (in time steps) we evaluate, (default=5000)")
    parser.add_argument("--max_timesteps", default=1e6, type=int, help="Max time steps to run environment (default=1 million)")
    parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise (default=0.1)")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic, (default is 256)")
    parser.add_argument("--discount", default=0.99, help="The discount factor (default=0.99)")
    parser.add_argument("--tau", default=0.005, help="Target network update rate (default=.005)")
    parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update (default=0.2)")
    parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--save_model", action="store_true", help="Save model and optimizer parameters")
    parser.add_argument("--load_model", default="", help="load the saved model")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0 
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                     ).clip(-max_action, max_action)

        # Permorm action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        #store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            #Reset environment
            state, done = env.reset(), False
            episode_reward += reward
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
            
