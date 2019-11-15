import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import gym
import numpy as np
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


def evaluate(model, num_steps=1000, render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)

        if render: env.render()

    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:",
          len(episode_rewards))
    env.close()
    return mean_100ep_reward


if __name__ == '__main__':
    # Silence tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Make environment
    model_env_name = 'InvertedPendulumGravity50-v2'
    env_name = 'InvertedPendulumGravity50-v2'
    env = gym.make('custom_envs:{}'.format(env_name))
    env = DummyVecEnv([lambda: env])
    env.reset()

    # Load agents
    # random_agent = PPO2(MlpPolicy, env)
    saved_ppo = PPO2.load('models/{}/{}.zip'.format(model_env_name,
                                                'PPO2_2019-11-14_161834.pkl'))

    # Evaluate
    print('Saved PPO:')
    evaluate(saved_ppo, num_steps=1000, render=True)
