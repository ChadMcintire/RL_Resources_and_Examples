import torch
import time

# Update the target network weights as is typical for actor critic models
# parameters is the weights of each
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.copy_(param.data)
        #this was the original code, the use of .data.copy_ can create problems
        #as per https://discuss.pytorch.org/t/using-tensor-data/79640
        #target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def validation_episodes(env, i_episodes, agent, writer):
    avg_reward = 0.
    episodes = 10

    for _ in range(episodes):
        state = env.reset()
        env.render()
        episode_reward = 0
        done = False
        render = True
        while not done:
            if render == True:
                env.render()
                time.sleep(.1)
            action = agent.select_action(state, evaluation=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes

    writer.add_scalar("avg_reward/test", avg_reward, i_episodes)

    print("-------------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {} Test Episode Number for the next 10{}".format(episodes, round(avg_reward, 2), i_episodes))
    print("----------------------------------------")

