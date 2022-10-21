from time import time
from env.TMNFEnv import TrackmaniaEnv
from agents.agent import RandomAgent
import time


def play_simulation():
    env = TrackmaniaEnv(action_space="controller")
    agent = RandomAgent()
    observation = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.act(observation)
        time.sleep(0.2)
        observation, reward, done, info = env.step(action)
        step += 1
        print(step)
        env.render()


if __name__ == "__main__":
    while True:
        play_simulation()   
