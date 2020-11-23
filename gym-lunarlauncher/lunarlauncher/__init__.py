from gym.envs.registration import register

register(
    id='lunarlauncher-v0',
    entry_point='gym_lunarlauncher.envs:LunarlauncherEnv',
)
register(
    id='lunarlauncher-extrahard-v0',
    entry_point='gym_lunarlauncher.envs:LunarlauncherExtraHardEnv',
)