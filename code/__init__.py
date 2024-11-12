from gymnasium import register

# Not sure if this is the correct way to register the environment as a gymnasium environment

register(
    id="fireEvac/PyroRL-v0",
    entry_point="code:WildfireEvacuationEnv",
    max_episode_steps=200,
)