import gym
from stable_baselines import PPO1, results_plotter
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import nail_bot
import numpy as np
import os
import tensorflow as tf

best_mean_reward, n_steps = -np.inf, 0

def callback(lcl, glb):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 1000 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x)>0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print('Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}'.format(best_mean_reward, mean_reward))
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print('Saving new best model')
                lcl['self'].save(log_dir + 'best_model.pk1')
    n_steps += 1
    return True

#Create log directory
log_dir = 'tmp/'
os.makedirs(log_dir, exist_ok = True)

#create custom policy- MLP Policy isn't big enough

policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[256,256,256,256])

#Create and train the model
env = gym.make('nailbot-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
model = PPO1(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
try:
    model = PPO1.load('ppo1_nailbot', env = env)
except:
    print("Could not find ppo1_nailbot file, creating new one")
total_timesteps = 50000
model.learn(total_timesteps=total_timesteps)
#save model
model.save('ppo1_nailbot')
print("Done training")

#show plot results
results_plotter.plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, 'PPO1 Nailbot')
plt.show()

        
'''
if __name__=='__main__':
        main()
'''
