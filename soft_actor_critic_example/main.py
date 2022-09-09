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
    writer = SummaryWriter(args.logs_dir + "/{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, 
                           args.policy, "autotune" if args.automatic_entropy_tuning else ""))


    #Memory replay, this helps with non-stationarity and the to help
    #the model not be sequential data for the for the Markov assumption
    memory = ReplayMemory(args.replay_size, args.seed)

    #Training Loop
    total_numsteps = 0
    updates = 0

    for i_episodes in itertools.count(1):
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
            current_reward = validation_episodes(env, total_num_steps, agent, writer, args.render)
            writer.add_scalar("reward/train", current_reward, total_num_steps)

            if max_reward <= current_reward:
                agent.save_checkpoint(args.env_name, suffix=total_numsteps, ckpt_path=args.check_pt_name)
                max_reward = current_reward
    
    env.close()
    writer.flush()
    writer.close()

    print("successfully exited")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch Soft Actor-Critic Args")
    parser.add_argument("--env-name", default="HalfCheetah-v2", help="Mujoco Gym environment (default: HalfCheetah-v20")
    parser.add_argument("--seed", type=int, default=123456, metavar="N", help="random seed (default: 123456)")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="G", help="discount facto for reward (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005, metavar="G", help="target smoothing coefficient(tau) (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.2, metavar="G", help="Temperature parameter alpha determines the relative importance of the entropy term against the reward (default: 0.2)")
    parser.add_argument("--policy", default="Gaussian", help="Policy Type: Gaussian | Determinitst (default: Gaussian)")
    parser.add_argument("--target_update_interval", type=int, default=1, metavar="N", help="value target update per no. of updates per step (default 1)")
    parser.add_argument("--automatic_entropy_tuning", action="store_true", dest="automatic_entropy_tuning", help="Automatically adjust alpha (default: False)")
    parser.add_argument("--no-automatic_entropy_tuning", action="store_false", dest="automatic_entropy_tuning", help="Automatically adjust alpha (default: False)")
    parser.set_defaults(render=False)

    parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument("--lr", type=float, default=0.0003, metavar="G", help="learning_rate (default: 0.0003)")
    parser.add_argument("--replay_size", type=int, default=1000000, metavar="N", help="size of replay buffer (default: 1000000)")
    parser.add_argument("--start_steps", type=int, default=10000, metavar="N", help="Steps sampling random action (default: 10000)")
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N', help='maximum number of steps (default: 1000000)')
    parser.add_argument('--eval', type=bool, default=True, help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step (default: 1)')
    parser.add_argument('--render', dest="render", action='store_true', help='Allows you to render the environment or not (defaults: False)')
    parser.add_argument('--no-render', dest="render", action='store_false', help='Allows you to render the environment or not (defaults: False)')
    parser.set_defaults(render=False)
    parser.add_argument('--steps_between_validation', type=int, default=10000, metavar='N', help='This gives the variable that allows the user to validate how the algorithm is performing')
    parser.add_argument("--check_pt_name", type=str, default="checkpoints", help="the name of the checkpoint path you want to use")
    parser.add_argument("--logs_dir", type=str, default="checkpoints", help="the name of the checkpoint path you want to use")

    args = parser.parse_args()

    training_loop(args)

