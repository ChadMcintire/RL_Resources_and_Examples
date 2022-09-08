import torch
import time
import glfw
import beepy

"""
*******************************************************************
* Update the target network weights as is typical for actor critic models
* parameters is the weights of each
*
* In this implementation this is only used to copy the initial of the critic 
* to the weights of the critic target
*
* for reference: https://discuss.pytorch.org/t/copy-weights-only-from-a-networks-parameters/5841
*******************************************************************
"""
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


"""
*******************************************************************
* Show 10 episodes under the current policy, this will render
* the environment for those episodes if the render variable is not
* false. This will also print the average reward.
*******************************************************************
"""
def validation_episodes(env, i_episodes, agent, writer, render):
    print("beginning validation")
    beepy.beep(sound=6)

    avg_reward = 0.
    episodes = 10

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
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

    #terminate window when not necessary
    #print("terminating window after validation")
    #neither close nor terminate work properly to close the glfw window and resume
    #after
    #env.close()
    #glfw.terminate()
    print("-------------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {} Test Episode Number for the next 10{}".format(episodes, round(avg_reward, 2), i_episodes))
    print("----------------------------------------")

    return avg_reward
