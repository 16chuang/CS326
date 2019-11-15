import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import gym
from datetime import datetime
import pytz
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# Silence tf warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

env_name = 'CartPole-v1'
env = gym.make(env_name)
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: env])

run_name = '{}_{}'.format(
    'PPO2',
    datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d_%H%M%S"))

model = PPO2(MlpPolicy,
             env,
             verbose=0,
             tensorboard_log='./logs/{}/{}'.format(env_name, run_name))
model.learn(total_timesteps=20000)
model.save('models/{}/{}'.format(env_name, run_name))

done = False
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("Episode finished early after {} timesteps".format(i + 1))
        break
    env.render()
if not done: print("Episode completed after all timesteps")

env.close()