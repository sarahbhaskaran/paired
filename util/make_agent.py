from algos import PPO, RolloutStorage, ACAgent, PlatoonACAgent, NonRecurrentRolloutStorage
from models import \
    MultigridNetwork, \
    MiniHackAdversaryNetwork, \
    NetHackAgentNet, \
    PlatoonNet
from trajectory.setup.setup_env import setup_env
from trajectory.setup.setup_exp import run_experiment
from trajectory.algos.ppo.policies import PopArtActorCriticPolicy

def model_for_multigrid_agent(
    env,
    agent_type='agent',
    recurrent_arch=None,
    recurrent_hidden_size=256):

    if 'adversary_env' in agent_type:
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space
        adversary_max_timestep = adversary_observation_space['time_step'].high[0] + 1
        adversary_random_z_dim = adversary_observation_space['random_z'].shape[0]

        model = MultigridNetwork(
            observation_space=adversary_observation_space,
            action_space=adversary_action_space,
            conv_filters=128,
            scalar_fc=10,
            scalar_dim=adversary_max_timestep,
            random_z_dim=adversary_random_z_dim,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        num_directions = observation_space['direction'].high[0] + 1

        model_constructor = MultigridNetwork
        model = model_constructor(
            observation_space=observation_space,
            action_space=action_space,
            scalar_fc=5,
            scalar_dim=num_directions,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)

    return model


def model_for_minihack_agent(
        env,
        agent_type='agent',
        recurrent_arch=None,
        recurrent_hidden_size=256,
    ):
    if 'adversary_env' in agent_type:
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space
        adversary_max_timestep = adversary_observation_space['time_step'].high[0] + 1
        adversary_random_z_dim = adversary_observation_space['random_z'].shape[0]

        model = MiniHackAdversaryNetwork(
                    observation_space=adversary_observation_space,
                    action_space=adversary_action_space,
                    recurrent_arch=recurrent_arch,
                    scalar_fc=10,
                    scalar_dim=adversary_max_timestep,
                    random_z_dim=adversary_random_z_dim,
                    obs_key='image')
    else:
        observation_space = env.observation_space
        action_space = env.action_space

        model = NetHackAgentNet(
            observation_shape=observation_space,
            num_actions = action_space.n,
            rnn_hidden_size = recurrent_hidden_size
        )

    return model

def platoon_agent(env, agent_type='agent'):
    # if 'adversary_env' in agent_type:
    #     adversary_observation_space = env.adversary_observation_space
    #     adversary_action_space = env.adversary_action_space
    #     # num_actions = env.num_actions
    #     adversary_max_timestep = adversary_observation_space['time_step'].high[0] + 1
    #     model = PlatoonNet(
    #                         observation_space=adversary_observation_space,
    #                         action_space=adversary_action_space
    #     )
    #
    # else:
    #     observation_space = env.observation_space
    #     action_space = env.action_space
    #     model = PlatoonNet(
    #                 observation_space=observation_space,
    #                 action_space=action_space,
    #                 env=env
    #     )
    #
    # return model


    configs = setup_env(args=PlatoonArgs())
    if len(configs) > 1:
        print('Unexpected length of configs')
        print(configs)
    algorithm, train_config, learn_config = run_experiment(configs[0])
    # We made the env out here so set it in train_config
    train_config['env'] = env
    # Algorithm should actually be different for agent type
    train_config['monitor_wrapper'] = False
    model = algorithm(**train_config)
    return PlatoonACAgent(algo=model, storage=None, learn_config=learn_config, agent_type=agent_type)


def model_for_env_agent(
    env_name,
    env,
    agent_type='agent',
    recurrent_arch=None,
    recurrent_hidden_size=256,
    use_skip=False,
    choose_start_pos=False,
    use_popart=False,
    adv_use_popart=False,
    use_categorical_adv=False,
    use_goal=False,
    num_goal_bins=1):
    assert agent_type in [
        'agent',
        'adversary_agent',
        'adversary_env']

    if env_name.startswith('MultiGrid'):
        model = model_for_multigrid_agent(
            env=env,
            agent_type=agent_type,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)
    elif env_name.startswith('MiniHack'):
        model = model_for_minihack_agent(
            env=env,
            agent_type=agent_type,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)
    # PLATOON
    elif env_name.startswith('Platoon'):
        model = model_for_platoon_agent(
            env=env,
            agent_type=agent_type
        )
    else:
        raise ValueError(f'Unsupported environment {env_name}.')

    return model


# PLATOON: Where is this called from?
def make_agent(name, env, args, device='cpu'):
    if args.env_name.startswith('Platoon'):
        # Agent storage added in just a bit!
        agent = platoon_agent(env=env, agent_type=name)

        storage = None
        is_adversary_env = 'env' in name
        if is_adversary_env:
            observation_space = env.adversary_observation_space
            action_space = env.adversary_action_space
            num_steps = observation_space['time_step'].high[0]
            entropy_coef = args.adv_entropy_coef
            ppo_epoch = args.adv_ppo_epoch
            num_mini_batch = args.adv_num_mini_batch
            max_grad_norm = args.adv_max_grad_norm
            use_popart = vars(args).get('adv_use_popart', False)
        else:
            observation_space = env.observation_space
            action_space = env.action_space
            num_steps = args.num_steps
            entropy_coef = args.entropy_coef
            ppo_epoch = args.ppo_epoch
            num_mini_batch = args.num_mini_batch
            max_grad_norm = args.max_grad_norm
            use_popart = vars(args).get('use_popart', False)

        use_proper_time_limits = \
            env.get_max_episode_steps() is not None and vars(args).get('handle_timelimits', False)

        storage = NonRecurrentRolloutStorage(
            model=agent.algo,
            num_steps=num_steps,
            num_processes=args.num_processes,
            observation_space=observation_space,
            action_space=action_space,
            use_proper_time_limits=use_proper_time_limits,
            use_popart=use_popart
        )
        agent.storage = storage
        return agent #somehow. trajectory-training-icra doesn't seem to have this at all
    # Create model instance
    is_adversary_env = 'env' in name

    if is_adversary_env:
        observation_space = env.adversary_observation_space
        action_space = env.adversary_action_space
        num_steps = observation_space['time_step'].high[0]
        recurrent_arch = args.recurrent_adversary_env and args.recurrent_arch
        entropy_coef = args.adv_entropy_coef
        ppo_epoch = args.adv_ppo_epoch
        num_mini_batch = args.adv_num_mini_batch
        max_grad_norm = args.adv_max_grad_norm
        use_popart = vars(args).get('adv_use_popart', False)
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        num_steps = args.num_steps
        recurrent_arch = args.recurrent_agent and args.recurrent_arch
        entropy_coef = args.entropy_coef
        ppo_epoch = args.ppo_epoch
        num_mini_batch = args.num_mini_batch
        max_grad_norm = args.max_grad_norm
        use_popart = vars(args).get('use_popart', False)

    recurrent_hidden_size = args.recurrent_hidden_size

    actor_critic = model_for_env_agent(
        args.env_name, env, name,
        recurrent_arch=recurrent_arch,
        recurrent_hidden_size=recurrent_hidden_size,
        use_skip=vars(args).get('use_skip', False),
        use_popart=vars(args).get('use_popart', False),
        adv_use_popart=vars(args).get('adv_use_popart', False))

    algo = None
    storage = None
    agent = None

    use_proper_time_limits = \
        env.get_max_episode_steps() is not None and vars(args).get('handle_timelimits', False)

    if args.algo == 'ppo':
        # Create PPO
        algo = PPO(
            actor_critic=actor_critic,
            clip_param=args.clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=max_grad_norm,
            clip_value_loss=args.clip_value_loss,
            log_grad_norm=args.log_grad_norm
        )

        # Create storage
        storage = RolloutStorage(
            model=actor_critic,
            num_steps=num_steps,
            num_processes=args.num_processes,
            observation_space=observation_space,
            action_space=action_space,
            recurrent_hidden_state_size=args.recurrent_hidden_size,
            recurrent_arch=args.recurrent_arch,
            use_proper_time_limits=use_proper_time_limits,
            use_popart=use_popart
        )

        agent = ACAgent(algo=algo, storage=storage).to(device)

    else:
        raise ValueError(f'Unsupported RL algorithm {algo}.')

    return agent


class PlatoonArgs:
    def __init__(self):
        # exp params
        self.expname = 'Platoon-v0'
        self.logdir ='./log' # 'Experiment logs, checkpoints and tensorboard files will be saved under {logdir}/{expname}_[current_time]/.')
        self.n_processes = 1 # 'Number of processes to run in parallel. Useful when running grid searches.''Can be more than the number of available CPUs.')
        self.s3 = False #'If set, experiment data will be uploaded to s3://trajectory.env/. ''AWS credentials must have been set in ~/.aws in order to use this.')

        self.iters = 800 # 'Number of iterations (rollouts) to train for.''Over the whole training, {iters} * {n_steps} * {n_envs} environment steps will be sampled.')
        # TODO: should be much more than this for convergence
        self.n_steps=640 #'Number of environment steps to sample in each rollout in each environment.''This can span over less or more than the environment horizon.''Ideally should be a multiple of {batch_size}.')
        self.n_envs=1 #'Number of environments to run in parallel.')

        self.cp_frequency=10 #'A checkpoint of the model will be saved every {cp_frequency} iterations.' 'Set to None to not save no checkpoints during training.'     'Either way, a checkpoint will automatically be saved at the end of training.')
        self.eval_frequency=10 # 'An evaluation of the model will be done and saved to tensorboard every {eval_frequency} iterations.' 'Set to None to run no evaluations during training.' 'Either way, an evaluation will automatically be done at the start and at the end of training.')
        # TODO: this will need to be changed eventually in order to evaluate but fine without for now
        self.no_eval=True# 'If set, no evaluation (ie. tensorboard plots) will be done.')

        # training params
        self.algorithm='PPO'#'RL algorithm to train with. Available options: PPO, TD3.')

        self.hidden_layer_size=64#'Hidden layer size to use for the policy and value function networks.'         'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')
        self.network_depth=4#'Number of hidden layers to use for the policy and value function networks.''The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')

        self.lr=3e-4
        # TODO: should be 5120 for convergence, I think
        self.batch_size=64#'Minibatch size.')
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

        self.is_paired = True
