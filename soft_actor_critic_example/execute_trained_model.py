import gym


def run(args):
    print("beginning runs")
    env = gym.make(args.env_name)

    #set random seeds, if low result the environment didn't learn
    #env.seed(args.seed)
    #env.action_space.seed(args.seed)
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)

    #initialize the sac agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    load_checkpoint(self, ckpt_path, evaluate=False)
    while true:
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            env.render()
            time.sleep(.1)
            action = agent.select_action(state, evaluation=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
