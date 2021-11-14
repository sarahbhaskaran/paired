
import gym
from envs.registration import register as gym_register

env_list = []

# Copied from register.py in /envs/multigrid
def register(env_id, entry_point, reward_threshold=0.95, max_episode_steps=None):
  """Register a new environment with OpenAI gym based on id."""
  assert env_id.startswith("Platoon-")
  if env_id in env_list:
    del gym.envs.registry.env_specs[id]
  else:
    # Add the environment to the set
    env_list.append(id)

  kwargs = dict(
    id=env_id,
    entry_point=entry_point,
    reward_threshold=reward_threshold
  )

  if max_episode_steps:
    kwargs.update({'max_episode_steps':max_episode_steps})

  # Register the environment with OpenAI gym
  gym_register(**kwargs)


class PlatoonEnv:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step_adversary(self, action):
        pass

    def reset_agent(self):
        pass

    def step(self, actions):
        pass

    def __str__(self):
        pass


# Copied from multigrid.py
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

# TODO: change max_episode_steps if necessary
register('Platoon-v0', module_path+":PlatoonEnv", max_episode_steps=250)
