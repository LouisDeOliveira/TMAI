import torch
import torch.nn as nn
import torch.optim as optim
from tmai.env.TMNFEnv import TrackmaniaEnv
from tmai.training.utils import Buffer, Transition, total_reward, play_episode
from tmai.agents.DDPG_agent import DDPG_agent
import numpy as np


class DDPG_trainer:
    def __init__(
        self,
        batch_size=32,
        N_epochs=10,
    ):
        self.N_epochs = N_epochs
        self.batch_size = batch_size
        self.GAMMA = 0.999
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.tau = 0.05

        self.env = TrackmaniaEnv(action_space="controller", n_rays=64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DDPG_agent(
            observation_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.shape[0],
            hidden_size=256,
        )

        self.target = DDPG_agent(
            observation_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.shape[0],
            hidden_size=256,
        )

        self.actor_optimizer = optim.Adam(
            self.model.policy.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.model.value.parameters(), lr=self.critic_lr
        )

        self.critic_loss = nn.MSELoss()

        self.buffer = Buffer(capacity=10000)
        self.fill_buffer()

    def fill_buffer(self):
        while len(self.buffer) < 2 * self.batch_size:
            episode = play_episode(
                self.model,
                self.env,
                act_value=lambda: [1.0, np.random.uniform(-1.0, 1.0)],
            )
            episode = filter(lambda transition: not transition.done, episode)
            self.buffer.append_multiple(list(episode))

    def optimization_step(self):
        batch = self.buffer.sample(self.batch_size)

        state_batch = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float
        ).to(self.device)
        action_batch = torch.tensor(
            np.array([t.action for t in batch]), dtype=torch.int64
        ).to(self.device)
        reward_batch = torch.tensor(
            np.array([t.reward for t in batch]), dtype=torch.float
        ).to(self.device)
        next_state_batch = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float
        ).to(self.device)
        print("reward_batch", reward_batch.shape)
        next_q = self.target.value(
            next_state_batch, self.target.policy(next_state_batch)
        )
        print("next_q", next_q.shape)
        target_q = reward_batch + self.GAMMA * next_q.squeeze()

        # update value network
        self.model.value.zero_grad()
        q_batch = self.model.value(state_batch, action_batch).squeeze()
        print("target_q", target_q.shape)
        print("q_batch", q_batch.shape)
        value_loss = self.critic_loss(q_batch, target_q)
        value_loss.backward()
        self.actor_optimizer.step()

        # update policy network
        self.model.policy.zero_grad()
        policy_loss = -self.model.value(
            state_batch, self.model.policy(state_batch).squeeze()
        ).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        self.update_target()

    def update_target(self):
        for target_param, param in zip(
            self.target.policy.parameters(), self.model.policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target.value.parameters(), self.model.value.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train(self):
        for _ in range(self.N_epochs):
            episode = []
            observation = self.env.reset()
            done = False
            step = 0
            while not done:
                prev_obs = observation
                action = self.model.act(observation)
                print(action)
                observation, reward, done, info = self.env.step(action)
                transition = Transition(prev_obs, action, observation, reward, done)
                episode.append(transition)
                step += 1
                self.optimization_step()

            self.buffer.append_multiple(episode)
            print("Episode reward: ", total_reward(episode))
            print("Episode length: ", len(episode))
        print("done")


if __name__ == "__main__":
    trainer = DDPG_trainer(N_epochs=1000)
    trainer.train()
