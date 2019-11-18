import os
from rl_with_teachers.teachers.base import TeacherPolicy

from stable_baselines import PPO2


class PPO2Agent(TeacherPolicy):
    def __init__(self, suffix):

        self.model = PPO2.load(
            '{}/trained_models/InvertedPendulum-v2/PPO2_{}.pkl'.format(
                os.path.dirname(__file__), suffix))

    def __call__(self, obs):
        action, _states = self.model.predict(obs)
        action /= 6.0
        return action
