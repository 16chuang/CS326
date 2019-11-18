from gym.envs.registration import register

register(
    id='InvertedPendulumDefault-v2',
    entry_point='custom_envs.envs:InvertedPendulumCustomEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedPendulumGravity1-v2',
    entry_point='custom_envs.envs:InvertedPendulumGravity1Env',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedPendulumGravity5-v2',
    entry_point='custom_envs.envs:InvertedPendulumGravity5Env',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedPendulumGravity20-v2',
    entry_point='custom_envs.envs:InvertedPendulumGravity20Env',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedPendulumGravity50-v2',
    entry_point='custom_envs.envs:InvertedPendulumGravity50Env',
    max_episode_steps=1000,
    reward_threshold=950.0,
)
