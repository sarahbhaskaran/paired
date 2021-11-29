import torch

class ACAgent(object):
	def __init__(self, algo, storage):
		self.algo = algo
		self.storage = storage

	def update(self):
		info = self.algo.update(self.storage)
		self.storage.after_update()

		return info

	def to(self, device):
		self.algo.actor_critic.to(device)
		self.storage.to(device)

		return self

	def train(self):
		self.algo.actor_critic.train()

	def eval(self):
		self.algo.actor_critic.eval()

	def random(self):
		self.algo.actor_critic.random = True

	def process_action(self, action):
		if hasattr(self.algo.actor_critic, 'process_action'):
			return self.algo.actor_critic.process_action(action)
		else:
			return action

	def act(self, *args, **kwargs):
		return self.algo.actor_critic.act(*args, **kwargs)

	def get_value(self, *args, **kwargs):
		return self.algo.actor_critic.get_value(*args, **kwargs)

	def insert(self, *args, **kwargs):
		return self.storage.insert(*args, **kwargs)

class PlatoonACAgent(object):
	def __init__(self, algo, storage, learn_config, agent_type):
		self.algo = algo
		self.storage = storage
		self.learn_config = learn_config
		self.agent_type = agent_type

	def update(self):
		info = self.algo.train(self.storage)
		self.storage.after_update()

		return info

	def to(self, device):
		self.algo.actor_critic.to(device)
		self.storage.to(device)

		return self

	def train(self):
		# self.algo.learn(**self.learn_config)
		pass

	def eval(self):
		self.algo.actor_critic.eval()

	def random(self):
		self.algo.actor_critic.random = True

	def process_action(self, action):
		if hasattr(self.algo.policy, 'process_action'):
			return self.algo.policy.process_action(action)
		else:
			return action

	def act(self, obs):
		if 'env' in self.agent_type:
			# Is there any call for saying whether it's deterministic or random? multigrid_models takes random as an initialization parameter
			# Also do we not use the timestep at all?
			traj_obs = obs['trajectory_obs']
			return self.algo.policy.forward(traj_obs)
		else:
			return self.algo.policy.forward(obs)

	def get_value(self, obs, **kwargs):
		if 'env' in self.agent_type:
			traj_obs = obs['trajectory_obs']
			return self.algo.policy.value_net(self.algo.policy.vf_extractor(traj_obs))
		return self.algo.policy.value_net(self.algo.policy.vf_extractor(obs))

	def insert(self, *args, **kwargs):
		return self.storage.insert(*args, **kwargs)
