import gym
from stable_baselines3 import PPO


if __name__ == "__main__":
    env = gym.make("CarRacing-v1")
    obs = env.reset()

    model = PPO.load("ppo_car_racing_custom", env)

    while True:
        action, _states = model.predict(obs.copy())
        obs, rewards, dones, info = env.step(action)
        env.render()
