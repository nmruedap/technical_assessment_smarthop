import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
import os.path
from os import path

#env_id = 'LunarLander-v2'
#video_folder = 'videos/'
env = gym.make('LunarLander-v2')

if path.isfile("l_lander_dqn.zip"):
    #env = DummyVecEnv([lambda: gym.make(env_id)])
    #env = VecVideoRecorder(env, video_folder,
    #                       record_video_trigger=lambda x: x == 0, video_length=1000,
    #                       name_prefix="lunar_lander_testing_agent")

    # Evaluation stage
    model = DQN.load("l_lander_dqn")
    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    #env.close()
else:
    # Learning stage
    model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

    # Train the agent
    model.learn(total_timesteps=int(5e5))
    model.save("l_lander_dqn")
    del model

    # Load the trained agent
    model = DQN.load("l_lander_dqn")

    # Evaluation environment
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()