import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import gym
from datetime import datetime
import pytz
import tensorflow as tf
import argparse

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DDPG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--timesteps',
                        type=int,
                        default='50000',
                        help='Num timesteps to train.')
    parser.add_argument('--env', type=str, default='', help='Gym environment.')
    parser.add_argument('--agent', type=str, default='', help='RL algorithm for agent')
    args = parser.parse_args()
    num_timesteps = args.__dict__['timesteps']
    env_name = args.__dict__['env']
    agent = args.__dict__['agent']

    # Silence tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    env = gym.make('custom_envs:{}'.format(env_name))
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: env])

    os.makedirs('./logs/{}'.format(env_name), exist_ok=True)
    os.makedirs('./models/{}'.format(env_name), exist_ok=True)
    run_name = '{}_{}'.format(
        'DDPG',
        datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d_%H%M%S"))

    if agent == 'DDPG':
        model = DDPG(DDPGMlpPolicy,
                     env,
                     verbose=0,
                     nb_train_steps=200,
                     nb_rollout_steps=100,
                     nb_eval_steps=200,
                     actor_lr=0.0005,
                     critic_lr=0.005,
                     memory_limit=1000000,
                     render_eval=True,
                     tensorboard_log='./logs/{}/{}'.format(env_name, run_name))
    elif agent == 'PPO':
        model = PPO2(MlpPolicy, env, verbose=0)
    print("Training starting...")
    model.learn(total_timesteps=num_timesteps)
    print("Training done.")
    model.save('./models/{}/{}'.format(env_name, run_name))

    ## Evaluate
    # done = False
    # obs = env.reset()
    # for i in range(100):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         print("Episode finished early after {} timesteps".format(i + 1))
    #         break
    #     env.render()
    # if not done: print("Episode completed after all timesteps")

    env.close()