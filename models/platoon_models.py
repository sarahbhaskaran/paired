from .common import *

from trajectory.setup.setup_env import setup_env
from trajectory.setup.setup_exp import run_experiment

class PlatoonNet(DeviceAwareModule):
    def __init__(self, **kwargs):
        configs = setup_env()
        algorithm, train_config, self.learn_config = run_experiment(configs)
        # Algorithm is PPO or TD3
        model = algorithm(train_config)

        # Stuff from multigrid_env

        # self.random = random
        # if isinstance(action_space, MultiDiscrete):
        #     self.num_actions = list(action_space.nvec)
        #     self.multi_dim = True
        #     self.action_dim = len(self.num_actions)
        #     self.num_action_logits = np.sum(list(self.num_actions))
        # else:
        #     self.num_actions = action_space.n
        #     self.multi_dim = False
        #     self.action_dim = 1
        #     self.num_action_logits = self.num_actions
        #
        # self.action_space = action_space
        #
        #
        # self.rnn = RNN(
        #     input_size=self.preprocessed_input_size,
        #     hidden_size=recurrent_hidden_size,
        #     arch=recurrent_arch)
        # self.base_output_size = recurrent_hidden_size
        #
        # self.actor = nn.Sequential(
        #     make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
        #     Categorical(actor_fc_layers[-1], self.num_actions)
        # )
        #
        # self.critic = nn.Sequential(
        #     make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.base_output_size),
        #     init_(nn.Linear(value_fc_layers[-1], 1))
        # )
        #
        # apply_init_(self.modules())

        self.train()

    @property
    def is_recurrent(self):
        return self.rnn is not None

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        if self.is_recurrent:
            return self.rnn.recurrent_hidden_state_size
        else:
            return 0

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _forward_base(self, inputs, rnn_hxs, masks):
        # Unpack input key values
        # TODO:

        in_embedded = torch.cat((in_image, in_x, in_y, in_scalar, in_z), dim=-1)

        core_features, rnn_hxs = self.rnn(in_embedded, rnn_hxs, masks)

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            B = inputs['image'].shape[0]
            action = torch.zeros((B,1), dtype=torch.int64, device=self.device)
            values = torch.zeros((B,1), device=self.device)
            action_log_dist = torch.ones(B, self.action_space.n, device=self.device)
            for b in range(B):
               action[b] = self.action_space.sample()

            return values, action, action_log_dist, rnn_hxs

        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)


        dist = self.actor(core_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        value = self.critic(core_features)

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        value = self.critic(core_features)
        return value, action_log_probs, dist_entropy, rnn_hxs


class PlatoonAgentNet:
    def __init__(self):
        pass
