import argparse
import gym
import torch
from sac import SAC
import numpy as np

def training_loop(args):
    #set up environment
    env = gym.make(args.env_name)

    #set up seeding for env
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    
    print("yes")















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch Soft Actor-Critic Args")
    parser.add_argument("--env-name", default="HalfCheetah-v2", help="Mujoco Gym environment (default: HalfCheetah-v20")
    parser.add_argument("--seed", type=int, default=123456, metavar="N", help="random seed (default: 123456)")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="G", help="discount facto for reward (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005, metavar="G", help="target smoothing coefficient(tau) (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.2, metavar="G", help="Temperature parameter alpha determines the relative importance of the entropy term against the reward (default: 0.2)")
    parser.add_argument("--policy", default="Gaussian", help="Policy Type: Gaussian | Determinitst (default: Gaussian)")
    parser.add_argument("--target_update_interval", type=int, default=1, metavar="N", help="value target update per no. of updates per step (default 1)")
    parser.add_argument("--automatic_entropy_tuning", type=bool, default=False, metavar="G", help="Automatically adjust alpha (default: False)")
    parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument("--lr", type=float, default=0.0003, metavar="G", help="learning_rate (default: 0.0003)")
    args = parser.parse_args()

    training_loop(args)

