import argparse
from training import training_loop
from execute_trained_model import run


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

    parser.add_argument('--train', dest="train", action='store_true', help='train a new model')
    parser.add_argument('--no-train', dest="train", action='store_false', help='load a previously trained model and run it')
    parser.add_argument("--QNet", type=str, default="original", help="original, other")
    args = parser.parse_args()

    if args.train:
        training_loop(args)
    else:
        run(args)

