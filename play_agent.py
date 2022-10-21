from time import time
from env.TMNFEnv import TrackmaniaEnv
from agents.agent import RandomAgent, RandomArrowsAgent
from env.TMIClient import ThreadedClient
import time


def play_simulation():
    env = TrackmaniaEnv(simthread=ThreadedClient(), action_space="arrows")
    agent = RandomArrowsAgent()
    observation = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.act(observation)
        print(action)
        observation, reward, done, info = env.step(action)
        step += 1
        print(step)
        env.render()


if __name__ == "__main__":
    while True:
        play_simulation()
