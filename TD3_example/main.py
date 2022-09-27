import argparse

from training import training_loop
from execute_trained_model import run

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
    parser.add_argument('--train', dest="train", action='store_true', help='train a new model')
    parser.add_argument('--no-train', dest="train", action='store_false', help='load a previously trained model and run it')
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Trainging: {args.train}")
    print("---------------------------------------")


    if args.train:
        training_loop(args)
    else:
        run(args)

