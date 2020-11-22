import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
import os.path
from os import path

if path.isfile("l_lander_dqn.zip"):
    # Evaluation stage
    model = DQN.load("l_lander_dqn")

    # Evaluation environment
    eval_env = gym.make('LunarLander-v2')
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    obs = eval_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        eval_env.render()
else:
    # Learning stage
    env = gym.make('LunarLander-v2')

    # DQN agent
    model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

    # Train the agent
    model.learn(total_timesteps=int(2e5))
    model.save("l_lander_dqn")
    del model

    # Load the trained agent
    model = DQN.load("l_lander_dqn")

    # Evaluation environment
    eval_env = gym.make('LunarLander-v2')
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    obs = eval_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        eval_env.render()