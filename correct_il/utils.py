#################################################################
# Utils
#################################################################
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import torch
import numpy as np
import random
from typing import Tuple
import pickle
import gym
from tqdm import tqdm

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from envs import *


def seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(trajectories: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.concatenate([traj["observations"][:-1] for traj in trajectories])
    a = np.concatenate([traj["actions"][:-1] for traj in trajectories])
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=-1)
    sp = np.concatenate([traj["observations"][1:] for traj in trajectories])
    return s, a, sp

def load_data(config_data):
    # TODO limit data number
    data_path = config_data.pkl
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print('loaded from pickle file...')
    return build_dataset(data)


# ============================================================================
# Flexible Data Loading System
# ============================================================================

class DataLoader:
    """
    Base class for data loaders. Subclass this to create custom data loaders.
    """
    def load(self, config_data):
        """
        Load data from the specified source.
        
        Args:
            config_data: Configuration object with data loading parameters
        
        Returns:
            Tuple of (observations, actions, next_observations) as numpy arrays
            For images: observations and next_observations are (N, H, W, C) arrays
            For states: observations and next_observations are (N, state_dim) arrays
        """
        raise NotImplementedError


class PickleTrajectoryLoader(DataLoader):
    """
    Loads data from pickle files containing trajectory lists.
    Each trajectory is a dict with "observations" and "actions" keys.
    """
    def load(self, config_data):
        data_path = config_data.pkl
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        print(f'Loaded {len(data)} trajectories from pickle file: {data_path}')
        return build_dataset(data)


class PickleImageLoader(DataLoader):
    """
    Loads image data from pickle files containing trajectory lists.
    Handles both 4D images and flattened images.
    """
    def load(self, config_data):
        data_path = config_data.pkl
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        images_list = []
        actions_list = []
        next_images_list = []
        
        image_shape = getattr(config_data, 'image_shape', None)
        
        for traj in data:
            obs = traj["observations"]
            acts = traj["actions"]
            
            # Check if observations are images (4D) or flattened
            if len(obs.shape) == 4:  # (T, H, W, C) - images
                images_list.append(obs[:-1])
                next_images_list.append(obs[1:])
                actions_list.append(acts[:-1])
            else:
                # Flattened images - need to reshape
                if image_shape is None:
                    raise ValueError("image_shape must be provided in config when observations are flattened")
                H, W, C = image_shape
                if obs.shape[1] != H * W * C:
                    raise ValueError(f"Flattened observation dimension {obs.shape[1]} doesn't match image_shape {image_shape}")
                obs_reshaped = obs.reshape(len(obs), H, W, C)
                images_list.append(obs_reshaped[:-1])
                next_images_list.append(obs_reshaped[1:])
                actions_list.append(acts[:-1])
        
        images = np.concatenate(images_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        next_images = np.concatenate(next_images_list, axis=0)
        
        print(f'Loaded {len(images)} image samples from {len(data)} trajectories')
        print(f'Image shape: {images.shape}, Action shape: {actions.shape}')
        
        return images, actions, next_images

class H5pyLoader(DataLoader):
    """
    Loads image data from h5py files.
    Returns a Dataset that can load data on-demand or into memory.
    """
    def load(self, config_data):
        from correct_il.datasets import H5pyImageDataset
        
        data_path = getattr(config_data, 'h5py_path', config_data.pkl)
        demo_prefix = getattr(config_data, 'demo_prefix', 'demo_')
        load_into_memory = getattr(config_data, 'load_into_memory', True)  # Default to True for speed
    
        action_horizon = int(getattr(config_data, 'action_horizon', 10))
        mask_weight = getattr(config_data, 'mask_weight', 4)  # Default mask_weight
      
        
        # Return a Dataset (load into memory by default for faster training)
        dataset = H5pyImageDataset(data_path, demo_prefix=demo_prefix, 
                                   load_into_memory=load_into_memory,
                                   action_horizon=action_horizon,
                                   mask_weight=mask_weight)
        print(f'Created H5pyImageDataset with {len(dataset)} samples')
        print(f'Image shape: {dataset.image_shape}, Action dim: {dataset.act_dim}')
        print(f'Load into memory: {load_into_memory}')
        print(f'Action horizon: {action_horizon} steps')
        
        return dataset


class DirectArrayLoader(DataLoader):
    """
    Loads data directly from numpy arrays or pre-processed data.
    Can load from file paths or use provided arrays.
    """
    def load(self, config_data):
        # Check if arrays are provided directly in config
        if hasattr(config_data, 'observations') and hasattr(config_data, 'actions'):
            obs = config_data.observations
            acts = config_data.actions
            
            # Handle both file paths and arrays
            if isinstance(obs, str):
                obs = np.load(obs)
            if isinstance(acts, str):
                acts = np.load(acts)
            
            # Create next observations
            next_obs = obs[1:]
            obs = obs[:-1]
            acts = acts[:-1]
            
            print(f'Loaded {len(obs)} samples from direct arrays')
            return obs, acts, next_obs
        
        # Try loading from specified files
        obs_path = getattr(config_data, 'observations_path', None)
        acts_path = getattr(config_data, 'actions_path', None)
        
        if obs_path and acts_path:
            obs = np.load(obs_path)
            acts = np.load(acts_path)
            next_obs = obs[1:]
            obs = obs[:-1]
            acts = acts[:-1]
            print(f'Loaded {len(obs)} samples from numpy files')
            return obs, acts, next_obs
        
        raise ValueError("DirectArrayLoader requires 'observations' and 'actions' in config, or 'observations_path' and 'actions_path'")


class CustomLoader(DataLoader):
    """
    Loads data using a custom function specified in config.
    """
    def load(self, config_data):
        loader_func = getattr(config_data, 'loader_func', None)
        if loader_func is None:
            raise ValueError("CustomLoader requires 'loader_func' in config")
        
        if isinstance(loader_func, str):
            # Import and call the function
            module_path, func_name = loader_func.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            loader_func = getattr(module, func_name)
        
        return loader_func(config_data)


# Data loader registry
DATA_LOADERS = {
    'pickle': PickleTrajectoryLoader,
    'pickle_images': PickleImageLoader,
    'h5py': H5pyLoader,
    'direct': DirectArrayLoader,
    'custom': CustomLoader,
}


def load_data_flexible(config_data):
    """
    Flexible data loading function that supports multiple formats and sources.
    
    Config options:
    - data.loader: 'pickle', 'pickle_images', 'h5py', 'direct', 'custom' (default: auto-detect)
    - data.pkl: Path to pickle file
    - data.h5py_path: Path to h5py file
    - data.image_shape: (H, W, C) for image data
    - data.use_images: bool, whether to use image loader
    - data.observations/actions: Direct numpy arrays
    - data.observations_path/actions_path: Paths to numpy files
    - data.loader_func: Custom loader function (string path or callable)
    
    Returns:
        Tuple of (observations, actions, next_observations)
    """
    # Determine loader type
    loader_type = getattr(config_data, 'loader', None)
    use_images = getattr(config_data, 'use_images', False)
    
    # Auto-detect if not specified
    if loader_type is None:
        if use_images:
            loader_type = 'pickle_images'
        elif hasattr(config_data, 'h5py_path'):
            loader_type = 'h5py'
        elif hasattr(config_data, 'observations') or hasattr(config_data, 'observations_path'):
            loader_type = 'direct'
        elif hasattr(config_data, 'loader_func'):
            loader_type = 'custom'
        else:
            loader_type = 'pickle'  # Default
    
    # Get loader class
    if loader_type not in DATA_LOADERS:
        raise ValueError(f"Unknown loader type: {loader_type}. Available: {list(DATA_LOADERS.keys())}")
    
    loader_class = DATA_LOADERS[loader_type]
    loader = loader_class()
    
    print(f'Using data loader: {loader_type}')
    return loader.load(config_data)

def dataset_to_d3rlpy(s, a, _):
    import d3rlpy
    rews = np.ones(len(s))
    terminals = np.zeros(len(s)) # Hack
    terminals[-1] = 1.0
    return d3rlpy.dataset.MDPDataset(
        observations=np.array(s, dtype=np.float32),
        actions=np.array(a, dtype=np.float32),
        rewards=np.array(rews, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
        episode_terminals=np.array(terminals, dtype=np.float32),
    )

def gen_noise_dataset(obs, noise):
    return obs + np.random.normal(0,noise,obs.shape)

def load_demo_for_policy(config):
    # Load Expert Data
    expert_dataset = dataset_to_d3rlpy(*load_data(config.data))
    if config.policy.naive:
        return expert_dataset, expert_dataset

    # if config.policy.noise_bc:
    #     full_dataset = dataset_to_d3rlpy(*load_data(config.data))
    #     s, a, sp = load_data(config.data)
    #     for _ in range(config.aug.num_labels):
    #         # import pdb;pdb.set_trace()
    #         new_s = gen_noise_dataset(s,0.0001)
    #         noise_dataset = dataset_to_d3rlpy(new_s,a,sp)

    #         full_dataset.extend(noise_dataset)
    #         full_dataset.extend(expert_dataset)

    #     return full_dataset, expert_dataset

    # Load Aug Data
    aug_pkl_fn = os.path.join(config.output.aug, 'aug_data.pkl')
    aug_data = pickle.load(open(aug_pkl_fn, 'rb'))
    _s = aug_data['observations']
    _original_s = aug_data['original_states']
    dis = np.linalg.norm(_s - _original_s, axis=1)
    sel_bounded = (dis < config.aug.epsilon) if config.aug.epsilon else np.ones_like(dis, dtype=bool)
    print(f'\033[93m Selected {sel_bounded.sum()} data points out of {len(_s)} \033[0m')
    aug_dataset = dataset_to_d3rlpy(
        aug_data['observations'][sel_bounded],
        aug_data['actions'][sel_bounded],
        aug_data['next_obs'][sel_bounded])



    # def batch_forward(f, states, actions):
    #     return [f(s.to('cpu').detach().numpy(),a.to('cpu').detach().numpy()) for s,a in zip(states, actions)]

    # env = gym.make('PendulumSwingupCont-v0')

    # accm_errors = []
    # for s,a,sp in zip(aug_data['observations'][sel_bounded],aug_data['actions'][sel_bounded],aug_data['next_obs'][sel_bounded]):
    #     gt_next = env.forward_solver(s,a)
    #     errors = sp - np.array(gt_next)
    #     errors = np.linalg.norm(errors)
    #     accm_errors.append(errors)

    # print(np.average(accm_errors))
    # import pdb;pdb.set_trace()
    # exit()

    aug_dataset.extend(expert_dataset)
    if config.aug.type == 'noisy_action' and config.aug.balance_data:
        for _ in range(config.aug.num_labels):
            aug_dataset.extend(expert_dataset)

    return aug_dataset, expert_dataset

def load_env(config):

    meta_env = False
    if config.env in ['hover-aviary-v0', 'flythrugate-aviary-v0', 'circle-aviary-v0']:
        import gym_pybullet_drones
        from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
        DEFAULT_OBS = ObservationType('kin')
        DEFAULT_ACT = ActionType('rpm')
        env = gym.make(f'{config.env}',
                            aggregate_phy_steps=1,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            gui=False
                            )
    elif config.env in ['assembly-v2', 'bin-picking-v2', 'button-press-topdown-v2', 'door-open-v2', 'drawer-close-v2', 'drawer-open-v2','handle-press-v2','plate-slide-v2',\
    'window-open-v2','disassemble-v2','faucet-open-v2','faucet-close-v2','coffee-button-v2','peg-unplug-side-v2','pick-place-wall-v2','reach-wall-v2',\
    'basketball-v2','coffee-pull-v2','coffee-push-v2','lever-pull-v2','pick-place-v2','push-v2','push-back-v2','reach-v2','soccer-v2','sweep-v2','window-close-v2']:

        import metaworld
        import metaworld.envs.mujoco.env_dict as _env_dict
        env = _env_dict.MT50_V2[config.env]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        meta_env = True
    else:
        env = gym.make(config.env)
    return env, meta_env

def evaluate_on_environment(
    env: gym.Env, algo, n_trials: int = 10, render: bool = False, metaworld=False,
    sensor_noise_size = None, actuator_noise_size=None):
    """Returns scorer function of evaluation on environment.
    """
    episode_rewards = []
    success = 0
    trail = 0
    drone_task = True if (env.spec is not None) and ('aviary' in env.spec.id) else False
    while True:
        observation = env.reset()
        if drone_task:
            observation = env.getDroneStateVector(0)
        episode_reward = 0.0
        steps = 0
        while True:

            # take action
            if sensor_noise_size:
                observation = observation * (1-sensor_noise_size) + sensor_noise_size * \
                    env.observation_space.sample() if not drone_task else \
                    np.random.uniform(low=-1.0, high=1.0, size=observation.shape)
            action = algo.predict([observation]) # no rescaling necessary
            if actuator_noise_size:
                action = action * (1-actuator_noise_size) + env.action_space.sample() * actuator_noise_size
            # action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            if drone_task:
                observation = env.getDroneStateVector(0)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
            if metaworld and steps > 499:
                break
            steps += 1
        success += info['success'] if 'success' in info else 0
        episode_rewards.append(episode_reward)
        trail+=1
        if trail >=n_trials:
            break

    return episode_rewards, success

class D3Agent():
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def load(self, model_folder, device):
        # load is handled at init
        pass
    # For 1-batch query only!
    def predict(self, sample):
        with torch.no_grad():
            input = torch.from_numpy(sample[0]).float().unsqueeze(0).to(self.device)
            at = self.policy(input)[0].cpu().numpy()
        return at

# ---------------------------------------------------------------
# Calculate EMD
# ---------------------------------------------------------------

def get_emd(X,Y) -> float:
    # get EMD/linear assignment
    #The points are arranged as m n-dimensional row vectors in the matrix X.
    d = cdist(X, Y)
    if np.inf in d:
        import pdb;pdb.set_trace()

    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / X.shape[0]


# ---------------------------------------------------------------
# Find root for equations
#

def root_finder_s(loss, s_next, a, init_s,
        max_iter=20,
        threshold=1e-4):
    # Find s as a solution to loss(s,a,s_next)
    # return s, is_success, # of iterations to solve

    s = init_s

    for i in range(max_iter):

        with torch.no_grad():
            r = loss(s, a, s_next)
            if torch.norm(r) <= threshold:
                break

        jac_s, _, _ = torch.autograd.functional.jacobian(
            loss, (s, a, s_next))

        delta_s = np.linalg.solve(
            jac_s.detach().cpu().numpy(),
            r.detach().cpu().numpy())
        s = s - torch.as_tensor(delta_s).to(s.device)

    if torch.norm(r) <= threshold:
        return s.detach(), 1, i

    return s.detach(), 0, max_iter


# ---------------------------------------------------------------
# Config Parser
# ---------------------------------------------------------------

import collections
import yaml
import os

def parse_config(parser):
    args, overrides = parser.parse_known_args()
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.update(vars(args))

    if not hasattr(config, 'debug'):
        config['debug'] = False

    if not hasattr(config, 'overwrite'):
        config['overwrite'] = 1

    config = parse_overrides(config, overrides)

    config['output']['location'] = os.path.expandvars(config['output']['location'])
    config['output']['location'] = os.path.join(config['output']['location'], f'seed{config["seed"]}')
    
    # Handle different data path types (pkl, h5py_path, etc.)
    if 'pkl' in config['data']:
        config['data']['pkl'] = os.path.expandvars(config['data']['pkl'])
    if 'h5py_path' in config['data']:
        config['data']['h5py_path'] = os.path.expandvars(config['data']['h5py_path'])
    if 'observations_path' in config['data']:
        config['data']['observations_path'] = os.path.expandvars(config['data']['observations_path'])
    if 'actions_path' in config['data']:
        config['data']['actions_path'] = os.path.expandvars(config['data']['actions_path'])
    
    config['output']['folder'] = os.path.join(
        config['output']['location'],
        f"{config['env']}{config['output']['folder_suffix']}")
    _folder = config['output']['folder']
    config = Namespace(config)

    # Generate intermediate filenames
    dynamics_optional = "" if config.dynamics.lipschitz_type == "none" else f"L{config.dynamics.lipschitz_constraint}"
    # Add action_horizon to output folder if specified
    action_horizon = getattr(config.data, 'action_horizon', None)
    action_horizon_suffix = f"_h{action_horizon}" if action_horizon and action_horizon != 1 else ""
    config['output']['dynamics'] = os.path.join(_folder, f"dynamics_{config.dynamics.lipschitz_type}{dynamics_optional}{action_horizon_suffix}")
    
    # Handle optional aug and policy sections
    if hasattr(config, 'aug') and hasattr(config.aug, 'type'):
        config['output']['aug'] = os.path.join(_folder, f"data/{config.aug.type}_{config.dynamics.lipschitz_type}{dynamics_optional}")
    else:
        config['output']['aug'] = os.path.join(_folder, f"data/default_{config.dynamics.lipschitz_type}{dynamics_optional}")
    
    if hasattr(config, 'policy'):
        if hasattr(config.policy, 'naive') and config.policy.naive:
            policy_folder = "policy/naive"
        elif hasattr(config, 'aug') and hasattr(config.aug, 'type'):
            policy_folder = f"policy/{config.aug.type}_{config.dynamics.lipschitz_type}{dynamics_optional}"
        else:
            policy_folder = f"policy/default_{config.dynamics.lipschitz_type}{dynamics_optional}"
        
        if hasattr(config.policy, 'noise_bc') and config.policy.noise_bc:
            policy_folder = f"policy/noise_{config.policy.noise_bc}"
        
        config['output']['policy'] = os.path.join(_folder, policy_folder)
    else:
        config['output']['policy'] = os.path.join(_folder, "policy/default")
    
    return config


def save_config_yaml(config, save_fn):
    data = dict(config)
    with open(save_fn, 'w') as out_file:
        yaml.dump(data, out_file, default_flow_style=False)


def parse_overrides(config, overrides):
    """
    Overrides the values specified in the config with values.
    config: (Nested) dictionary of parameters
    overrides: Parameters to override and new values to assign. Nested
        parameters are specified via dot notation.
    >>> parse_overrides({}, [])
    {}
    >>> parse_overrides({}, ['a'])
    Traceback (most recent call last):
      ...
    ValueError: invalid override list
    >>> parse_overrides({'a': 1}, [])
    {'a': 1}
    >>> parse_overrides({'a': 1}, ['a', 2])
    {'a': 2}
    >>> parse_overrides({'a': 1}, ['b', 2])
    Traceback (most recent call last):
      ...
    KeyError: 'b'
    >>> parse_overrides({'a': 0.5}, ['a', 'test'])
    Traceback (most recent call last):
      ...
    ValueError: could not convert string to float: 'test'
    >>> parse_overrides(
    ...    {'a': {'b': 1, 'c': 1.2}, 'd': 3},
    ...    ['d', 1, 'a.b', 3, 'a.c', 5])
    {'a': {'b': 3, 'c': 5.0}, 'd': 1}
    """
    if len(overrides) % 2 != 0:
        # print('Overrides must be of the form [PARAM VALUE]*:', ' '.join(overrides))
        raise ValueError('invalid override list')

    for param, value in zip(overrides[::2], overrides[1::2]):
        keys = param.split('.')
        params = config
        for k in keys[:-1]:
            if k not in params:
                raise KeyError(param)
            params = params[k]
        if keys[-1] not in params:
            raise KeyError(param)

        current_type = type(params[keys[-1]])
        value = current_type(value)  # cast to existing type
        params[keys[-1]] = value

    return config


# Namespace for command line overwriting of training
class Namespace(collections.abc.MutableMapping):
    """Utility class to convert a (nested) dictionary into a (nested) namespace.
    >>> x = Namespace({'foo': 1, 'bar': 2})
    >>> x.foo
    1
    >>> x.bar
    2
    >>> x.baz
    Traceback (most recent call last):
        ...
    KeyError: 'baz'
    >>> x
    {'foo': 1, 'bar': 2}
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': 1, 'bar': 2}
    >>> x = Namespace({'foo': {'a': 1, 'b': 2}, 'bar': 3})
    >>> x.foo.a
    1
    >>> x.foo.b
    2
    >>> x.bar
    3
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': {'a': 1, 'b': 2}, 'bar': 3}
    >>> (lambda **kwargs: print(kwargs))(**x.foo)
    {'a': 1, 'b': 2}
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, k):
        if not k.startswith('_'):
            if k not in self._data:
                return Namespace({})
            v = self._data[k]
            if isinstance(v, dict):
                v = Namespace(v)
            return v

        if k not in self.__dict__:
            raise AttributeError("'Namespace' object has no attribute '{}'".format(k))

        return self.__dict__[k]

    def __repr__(self):
        return repr(self._data)
