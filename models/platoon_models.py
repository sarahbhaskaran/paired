from .common import *

from trajectory.setup.setup_env import setup_env
from trajectory.setup.setup_exp import run_experiment

from gym.spaces import MultiDiscrete

#  NOT USED ANYMORE--THIS LOGIC JUST HAPPENS IN MAKE_AGENT.PY

class PlatoonNet(DeviceAwareModule):
    def __init__(self, observation_space, action_space, env, random=False, **kwargs):
        configs = setup_env(args=PlatoonArgs())
        if len(configs) > 1:
            print('Unexpected length of configs')
            print(configs)
        algorithm, train_config, self.learn_config = run_experiment(configs[0])
        # We made the env out here so set it in train_config
        train_config['env'] = env
        # Algorithm is PPO or TD3. But actually for now avoid making it in here since it will try to wrap the environment
        train_config['monitor_wrapper'] = False
        model = algorithm(**train_config)
        # policy = train_config['policy']

        # Stuff from multigrid_env

        # self.random = random
        # if isinstance(action_space, MultiDiscrete):
        #     self.num_actions = list(action_space.nvec)
        #     self.multi_dim = True
        #     self.action_dim = len(self.num_actions)
        #     self.num_action_logits = np.sum(list(self.num_actions))
        # else:
        #     # Maybe we don't have num_actions since action space is not discrete
        #     # self.num_actions = action_space.n
        #     self.multi_dim = False
        #     self.action_dim = 1
        #     # self.num_action_logits = self.num_actions
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
        #
        # self.train()

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

class PlatoonArgs:
    def __init__(self):
        # exp params
        self.expname = 'Platoon-v0'
        self.logdir ='./log' # 'Experiment logs, checkpoints and tensorboard files will be saved under {logdir}/{expname}_[current_time]/.')
        self.n_processes = 1 # 'Number of processes to run in parallel. Useful when running grid searches.''Can be more than the number of available CPUs.')
        self.s3 = False #'If set, experiment data will be uploaded to s3://trajectory.env/. ''AWS credentials must have been set in ~/.aws in order to use this.')

        self.iters = 800 # 'Number of iterations (rollouts) to train for.''Over the whole training, {iters} * {n_steps} * {n_envs} environment steps will be sampled.')
        self.n_steps=640 #'Number of environment steps to sample in each rollout in each environment.''This can span over less or more than the environment horizon.''Ideally should be a multiple of {batch_size}.')
        self.n_envs=1 #'Number of environments to run in parallel.')

        self.cp_frequency=10 #'A checkpoint of the model will be saved every {cp_frequency} iterations.' 'Set to None to not save no checkpoints during training.'     'Either way, a checkpoint will automatically be saved at the end of training.')
        self.eval_frequency=10 # 'An evaluation of the model will be done and saved to tensorboard every {eval_frequency} iterations.' 'Set to None to run no evaluations during training.' 'Either way, an evaluation will automatically be done at the start and at the end of training.')
        self.no_eval=False# 'If set, no evaluation (ie. tensorboard plots) will be done.')

        # training params
        self.algorithm='PPO'#'RL algorithm to train with. Available options: PPO, TD3.')

        self.hidden_layer_size=64#'Hidden layer size to use for the policy and value function networks.'         'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')
        self.network_depth=4#'Number of hidden layers to use for the policy and value function networks.''The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')

        self.lr=3e-4
        self.batch_size=5120#'Minibatch size.')
        self.n_epochs=10#'Number of SGD iterations per training iteration.')
        self.gamma=0.99#'Discount factor.')
        self.gae_lambda=0.99#' Factor for trade-off of bias vs. variance for Generalized Advantage Estimator.')

        self.augment_vf=1#'If true, the value function will be augmented with some additional states.')

        # env params
        self.env_num_concat_states=1#'This many past states will be concatenated. If set to 1, it\'s just the current state. ' 'This works only for the base states and not for the additional vf states.')
        self.env_discrete=0#'If true, the environment has a discrete action space.')
        self.use_fs=0#'If true, use a FollowerStopper wrapper.')
        self.env_include_idm_mpg=0#'If true, the mpg is calculated averaged over the AV and the 5 IDMs behind.')
        self.env_horizon=1000#'Sets the training horizon.')
        self.env_max_headway=120#'Sets the headway above which we get penalized.')
        self.env_minimal_time_headway=1.0#'Sets the time headway below which we get penalized.')
        self.env_num_actions=7#'If discrete is set, the action space is discretized by 1 and -1 with this many actions')
        self.env_num_steps_per_sim=1#'We take this many sim-steps per environment step i.e. this lets us taking steps bigger than 0.1')

        self.env_platoon='av human*5'#'Platoon of vehicles following the leader. Can contain either "human"s or "av"s. ' '"(av human*2)*2" can be used as a shortcut for "av human human av human human". ' 'Vehicle tags can be passed with hashtags, eg "av#tag" "human#tag*3"')
        self.env_human_kwargs='{}'#'Dict of keyword arguments to pass to the IDM platoon cars controller.')
