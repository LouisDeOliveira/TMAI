import torch
import torch.optim as optim
import torch.nn as nn
from tmai.env.TMNFEnv import TrackmaniaEnv
from tmai.training.utils import Buffer, Transition, play_episode
from tmai.agents.DQN_agent import EpsilonGreedyDQN
import numpy as np


class DQN_trainer:
    def __init__(self, batch_size=32, N_epochs=100):
        self.N_epochs = N_epochs
        self.GAMMA = 0.999
        self.target_update = 2
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = Buffer(capacity=10000)
        self.env = TrackmaniaEnv(action_space="arrows")
        self.agent = EpsilonGreedyDQN(
            input_size=self.env.observation_space.shape[0], device=self.device
        )
        self.optimizer = optim.Adam(self.agent.policy.parameters(), lr=0.001)
        self.loss = nn.SmoothL1Loss()
        self.fill_buffer()

    def fill_buffer(self):
        while len(self.buffer) < self.batch_size:
            episode = play_episode(self.agent, self.env)
            episode = filter(lambda transition: not transition.done, episode)
            self.buffer.append_episode(list(episode))

    def optimze_step(self):
        batch = self.buffer.sample(self.batch_size)
        inv_map = {tuple(v): k for k, v in self.agent.action_correspondance.items()}
        state_batch = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float
        ).to(self.device)
        action_batch = torch.tensor(
            np.array([inv_map[tuple(t.action)] for t in batch]), dtype=torch.int64
        ).to(self.device)
        reward_batch = torch.tensor(
            np.array([t.reward for t in batch]), dtype=torch.float
        ).to(self.device)
        next_state_batch = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float
        ).to(self.device)
        state_action_values = (
            self.agent.policy(state_batch)
            .gather(1, action_batch.unsqueeze(1))
            .squeeze(1)
        )
        # print(state_action_values.shape)
        next_state_values = self.agent.target(next_state_batch).max(1)[0].detach()
        # print(next_state_values.shape)
        # print(reward_batch.shape)
        expected_action_values = (next_state_values * self.GAMMA) + reward_batch
        # print(state_action_values.shape, expected_action_values.shape)
        loss = self.loss(state_action_values, expected_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for epoch in range(self.N_epochs):
            self.env.reset()
            episode = []
            observation = self.env.reset()
            done = False
            step = 0
            while not done:
                prev_obs = observation
                action = self.agent.act(observation)
                action[0] = 1
                action[1] = 0
                print(action)
                observation, reward, done, info = self.env.step(action)
                transition = Transition(prev_obs, action, observation, reward, done)
                episode.append(transition)
                step += 1
                self.env.render()
                self.optimze_step()

            self.buffer.append_episode(episode)
            if epoch % self.target_update == 0:
                self.agent.target.load_state_dict(self.agent.policy.state_dict())


if __name__ == "__main__":
    trainer = DQN_trainer(N_epochs=200)
    trainer.train()
