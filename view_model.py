import gym
import nail_bot
from stable_baselines import PPO1

env = gym.make('nailbot-v0')
model = PPO1.load('ppo1_nailbot', env = env)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render('human')
