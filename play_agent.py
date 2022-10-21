from agents.agent import RandomArrowsAgent
from env.TMIClient import ThreadedClient
from env.TMNFEnv import TrackmaniaEnv


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
