from stable_baselines3 import PPO

from tmai.env.TMNFEnv import TrackmaniaEnv

if __name__ == "__main__":
    env = TrackmaniaEnv(action_space="gamepad")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
