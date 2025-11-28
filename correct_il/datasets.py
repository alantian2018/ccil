"""
PyTorch Dataset classes for CCIL.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDynamicsDataset(Dataset):
    """
    Dataset for image-based dynamics model training.
    
    Args:
        images: (N, H, W, C) numpy array of current images
        actions: (N, act_dim) numpy array of actions
        next_images: (N, H, W, C) numpy array of next images
    """
    def __init__(self, images, actions, next_images):
        self.images = images
        self.actions = actions
        self.next_images = next_images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'images': self.images[idx],
            'actions': self.actions[idx],
            'next_images': self.next_images[idx]
        }


class TrainingDataset(Dataset):
    """
    Wrapper dataset that adds Y (normalized target) values to a base dataset.
    
    Args:
        base_dataset: Base dataset (e.g., ImageDynamicsDataset)
        Y: (N, latent_dim) numpy array or torch tensor of normalized targets
    """
    def __init__(self, base_dataset, Y):
        self.base_dataset = base_dataset
        self.Y = Y.numpy() if isinstance(Y, torch.Tensor) else Y  # convert to numpy
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item['Y'] = self.Y[idx]
        return item


class H5pyImageDataset(Dataset):
    """
    Dataset that loads images and actions directly from an h5py file.
    Can operate in two modes:
    - Lazy mode (default): Loads data on-demand from hdf5 (slower but memory efficient)
    - In-memory mode: Loads all data into memory at initialization (faster but uses more memory)
    
    Args:
        h5py_path: Path to h5py file
        demo_prefix: Prefix for demonstration keys (default: 'demo_')
        load_into_memory: If True, load all data into memory at initialization (default: False)
        action_horizon: Number of action steps to predict ahead (default: 1). 
                       If n > 1, returns actions[current:current+n] and next_image at current+n
        mask_weight: Weight multiplier for robot_mask regions (default: 4). 
                     If None and masks exist, weights will be 1.0 everywhere.
    """
    def __init__(self, h5py_path, demo_prefix='demo_', load_into_memory=False, action_horizon=10, mask_weight=4):
        import h5py
        from tqdm import tqdm
        
        self.h5py_path = h5py_path
        self.demo_prefix = demo_prefix
        self.load_into_memory = load_into_memory
        self.action_horizon = action_horizon
        self.mask_weight = mask_weight
        # Open file and get demo keys
        with h5py.File(h5py_path, 'r') as f:
            self.demo_keys = sorted([key for key in f['data'].keys() if key.startswith(demo_prefix)])
            
            # Pre-compute indices: (demo_idx, timestep_idx) for each sample
            self.indices = []
            for demo_idx, demo_key in enumerate(self.demo_keys):
                demo = f['data'][demo_key]
                if 'actions' not in demo or 'obs' not in demo:
                    continue
                obs_group = demo['obs']
                if 'agentview_image' not in obs_group:
                    continue
                
                acts_len = demo['actions'].shape[0]
                agentview_len = obs_group['agentview_image'].shape[0]
                min_len = min(acts_len, agentview_len)
                
                # Add (demo_idx, timestep_idx) for each valid timestep
                # Need to ensure we have enough steps ahead for action_horizon
                for t in range(min_len):
                    self.indices.append((demo_idx, t))
            
            # Get shapes from first demo
            # Check if robot_mask exists
            self.has_mask = False
            if len(self.demo_keys) > 0:
                demo = f['data'][self.demo_keys[0]]
                self.base_act_dim = demo['actions'].shape[1]  # Base action dimension (e.g., 7)
                # Flattened action dimension: action_horizon * base_act_dim
                self.act_dim = self.base_act_dim * self.action_horizon
                agentview_shape = demo['obs']['agentview_image'].shape
                self.image_shape = (agentview_shape[1], agentview_shape[2], agentview_shape[3])  # Only agentview
                # Check if robot_mask exists
                if 'obs' in demo and 'robot_mask' in demo['obs']:
                    self.has_mask = True
                    mask_shape = demo['obs']['robot_mask'].shape
                    self.mask_shape = (mask_shape[1], mask_shape[2])  # (H, W)
        
        # Load all data into memory if requested
        if load_into_memory:
            print(f"Loading {len(self.indices)} samples into memory from h5py file...")
            self.images = np.zeros((len(self.indices),) + self.image_shape, dtype=np.float32)
            self.next_images = np.zeros((len(self.indices),) + self.image_shape, dtype=np.float32)
            # Actions shape: (n_samples, flattened_act_dim) - always flattened
            self.actions = np.zeros((len(self.indices), self.act_dim), dtype=np.float32)
            # Masks for next images (corresponding to next_images)
            if self.has_mask:
                self.masks = np.zeros((len(self.indices),) + self.mask_shape, dtype=bool)
                # Pre-compute weights: (N, H, W) -> (N, H, W, 6)
                self.reconstruction_weights = None  # Will be computed after loading masks
            
            # Check if normalization is needed (check first sample)
            needs_normalization = False
            with h5py.File(h5py_path, 'r') as f:
                if len(self.indices) > 0:
                    demo_idx, timestep_idx = self.indices[0]
                    demo = f['data'][self.demo_keys[demo_idx]]
                    obs_group = demo['obs']
                    sample_img = obs_group['agentview_image'][timestep_idx]
                    if sample_img.max() > 1.0:
                        needs_normalization = True
            
            with h5py.File(h5py_path, 'r') as f:
                for idx, (demo_idx, timestep_idx) in enumerate(tqdm(self.indices, desc="Loading data")):
                    demo = f['data'][self.demo_keys[demo_idx]]
                    obs_group = demo['obs']
                    len_traj = demo['actions'].shape[0]
                    len_obs = obs_group['agentview_image'].shape[0]
                    last_timestep = min(len_traj, len_obs) - 1
                    
                    # Load current image
                    agentview_curr = obs_group['agentview_image'][timestep_idx]
                    images_curr = agentview_curr  # (H, W, 3)
                    
                    # Load next image at current + action_horizon or last available obs
                    next_timestep = min(timestep_idx + self.action_horizon, last_timestep)
                    agentview_next = obs_group['agentview_image'][next_timestep]
                    images_next = agentview_next  # (H, W, 3)
                    
                    # Load mask for next image if available
                    if self.has_mask:
                        mask = obs_group['robot_mask'][next_timestep]  # (H, W) boolean mask
                        self.masks[idx] = mask
                    
                    # Load action sequence: actions[current : current + action_horizon]
                    # Pad with zeros if there aren't enough timesteps
                    available_actions = demo['actions'][timestep_idx:timestep_idx + self.action_horizon]  # (n_available, base_act_dim)
                    if len(available_actions) < self.action_horizon:
                        # Pad with zeros to get exactly action_horizon actions
                        padding = np.zeros((self.action_horizon - len(available_actions), self.base_act_dim), dtype=np.float32)
                        actions = np.concatenate([available_actions, padding], axis=0)  # (action_horizon, base_act_dim)
                    else:
                        actions = available_actions.astype(np.float32)  # (action_horizon, base_act_dim)
                    # Flatten actions: (action_horizon, base_act_dim) -> (action_horizon * base_act_dim,)
                    actions = actions.flatten()
                    # Normalize images to [0, 1] if needed (do it once during load, not in __getitem__)
                    if needs_normalization:
                        images_curr = images_curr / 255.0
                        images_next = images_next / 255.0
                    
                    self.images[idx] = images_curr.astype(np.float32)
                    self.next_images[idx] = images_next.astype(np.float32)
                    self.actions[idx] = actions.astype(np.float32)
            
            # Compute reconstruction weights from masks if available
            if self.has_mask and self.mask_weight > 0:
                # Convert boolean masks to float: True -> 1.0, False -> 0.0
                mask_values = self.masks.astype(np.float32)  # (N, H, W)
                # Compute weights: 1 + mask_weight * mask_values
                weights = 1.0 + self.mask_weight * mask_values  # (N, H, W)
                # Expand to match image shape: (N, H, W) -> (N, H, W, 3)
                weights = np.expand_dims(weights, axis=-1)  # (N, H, W, 1)
                weights = np.repeat(weights, self.image_shape[2], axis=-1)  # (N, H, W, 3)
                self.reconstruction_weights = weights.astype(np.float32)
            elif self.has_mask or self.mask_weight == 0:
                # If mask_weight is None, use uniform weights
                print(f"Using uniform weights for reconstruction")
                self.reconstruction_weights = np.ones((len(self.indices),) + self.image_shape, dtype=np.float32)
            else:
                self.reconstruction_weights = None
            
            print(f"Loaded {len(self.images)} samples into memory")
            print(f"Memory usage: {self.images.nbytes / (1024**3):.2f} GB for images, "
                  f"{self.actions.nbytes / (1024**2):.2f} MB for actions")
            if needs_normalization:
                print("Images normalized to [0, 1] range during load")
    
    def __len__(self):
        return len(self.indices)
    
    def get_masks(self):
        """
        Get all masks as a numpy array.
        Returns None if masks are not available.
        """
        if not self.has_mask:
            return None
        if self.load_into_memory:
            return self.masks
        else:
            # Load all masks on demand
            import h5py
            masks = []
            with h5py.File(self.h5py_path, 'r') as f:
                for demo_idx, timestep_idx in self.indices:
                    demo = f['data'][self.demo_keys[demo_idx]]
                    obs_group = demo['obs']
                    len_traj = demo['actions'].shape[0]
                    len_obs = obs_group['agentview_image'].shape[0]
                    last_timestep = min(len_traj, len_obs) - 1
                    next_timestep = min(timestep_idx + self.action_horizon, last_timestep)
                    mask = obs_group['robot_mask'][next_timestep]
                    masks.append(mask)
            return np.array(masks, dtype=bool)
    
    def __getitem__(self, idx):
        if self.load_into_memory:
            # Return from pre-loaded arrays (already normalized)
            result = {
                'images': self.images[idx],
                'actions': self.actions[idx],
                'next_images': self.next_images[idx]
            }
            if self.reconstruction_weights is not None:
                result['reconstruction_weights'] = self.reconstruction_weights[idx]
            return result
        else:
            # Load on-demand from hdf5
            import h5py
            demo_idx, timestep_idx = self.indices[idx]
            
            with h5py.File(self.h5py_path, 'r') as f:
                demo = f['data'][self.demo_keys[demo_idx]]
                obs_group = demo['obs']
                len_traj = demo['actions'].shape[0]
                len_obs = obs_group['agentview_image'].shape[0]
                last_timestep = min(len_traj, len_obs) - 1
                
                # Load current image
                agentview_curr = obs_group['agentview_image'][timestep_idx]
                images_curr = agentview_curr  # (H, W, 3)
                
                # Load next image at current + action_horizon or last available obs
                next_timestep = min(timestep_idx + self.action_horizon, last_timestep)
                agentview_next = obs_group['agentview_image'][next_timestep]
                images_next = agentview_next  # (H, W, 3)
                
                # Load mask and compute weights if available
                weights = None
                if self.has_mask and self.mask_weight is not None:
                    mask = obs_group['robot_mask'][next_timestep]  # (H, W) boolean mask
                    # Convert to float and compute weights
                    mask_values = mask.astype(np.float32)  # (H, W)
                    weights = 1.0 + self.mask_weight * mask_values  # (H, W)
                    # Expand to match image shape: (H, W) -> (H, W, 3)
                    weights = np.expand_dims(weights, axis=-1)  # (H, W, 1)
                    weights = np.repeat(weights, images_next.shape[2], axis=-1)  # (H, W, 3)
                    weights = weights.astype(np.float32)
                
                # Load action sequence: actions[current : current + action_horizon]
                # Pad with zeros if there aren't enough timesteps
                available_actions = demo['actions'][timestep_idx:timestep_idx + self.action_horizon]  # (n_available, base_act_dim)
                if len(available_actions) < self.action_horizon:
                    # Pad with zeros to get exactly action_horizon actions
                    padding = np.zeros((self.action_horizon - len(available_actions), self.base_act_dim), dtype=np.float32)
                    actions = np.concatenate([available_actions, padding], axis=0)  # (action_horizon, base_act_dim)
                else:
                    actions = available_actions.astype(np.float32)  # (action_horizon, base_act_dim)
                # Flatten actions: (action_horizon, base_act_dim) -> (action_horizon * base_act_dim,)
                actions = actions.flatten()
            
            # Normalize images to [0, 1] if needed (only for lazy loading)
            images_curr = images_curr.astype(np.float32)
            images_next = images_next.astype(np.float32)
            if images_curr.max() > 1.0:
                images_curr = images_curr / 255.0
                images_next = images_next / 255.0
            
            result = {
                'images': images_curr,
                'actions': actions.astype(np.float32),
                'next_images': images_next
            }
            if weights is not None:
                result['reconstruction_weights'] = weights
            return result

