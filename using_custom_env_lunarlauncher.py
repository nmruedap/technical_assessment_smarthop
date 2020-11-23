import gym

from stable_baselines import DQN
import lunarlauncher_env
from os import path

env = lunarlauncher_env.LunarLauncherEnv(False)

if path.isfile("l_launcher_dqn.zip"):
    # Evaluation stage
    model = DQN.load("l_launcher_dqn")
    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
else:
    # Learning stage
    model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

    # Train the agent
    model.learn(total_timesteps=int(1e5))
    model.save("l_launcher_dqn")
    del model

    # Load the trained agent
    model = DQN.load("l_launcher_dqn")

    # Evaluation environment
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

'''
env = lunarlauncher_env.LunarLauncherEnv(False)
obs = env.reset()
for i in range(100):
    obs, rewards, dones, info = env.step(2)
    #if (i % 2) == 0:
    #    obs, rewards, dones, info = env.step(3)
    #else:
    #    obs, rewards, dones, info = env.step(2)
    env.render()
'''