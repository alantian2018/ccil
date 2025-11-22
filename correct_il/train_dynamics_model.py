"""
Script to learn MDP model from data for offline policy optimization
"""

import argparse
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np

from models.nn_dynamics import WorldModel, ImageWorldModel
from utils import seed, parse_config, load_data, load_data_flexible, save_config_yaml
import torch


def construct_parser():
    parser = argparse.ArgumentParser(description='Training Dynamic Functions.')
    parser.add_argument("config_path", help="Path to config file")
    return parser


def plot_loss(train_loss, fn, xbar=None, title='Train Loss'):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Dynamics Model Loss")
    ax.set_ylabel(title)
    ax.set_xlabel("Epoch")
    if xbar:
        ax.axhline(xbar, linestyle="--", color="black")
    ax.plot(train_loss)
    fig.savefig(fn)


def save_loss(train_loss, folder_name, prediction_error, model_name=None, eval_loss=None):
    model_name = f"{model_name}_" if model_name else ""
    for loss_name, losses in train_loss.items():
      fn_prefix = os.path.join(folder_name, f'{model_name}train_{loss_name}')
      plot_loss(losses, fn_prefix + '.png', title=loss_name)
      with open(fn_prefix+'.txt', 'w') as f:
        _l = np.array2string(np.array(losses), formatter={'float_kind':lambda x: "%.6f\n" % x})
        f.write(_l)
    with open(os.path.join(folder_name, f'{model_name}statistics.txt'), 'w') as f:
      if isinstance(prediction_error, dict):
          f.write(f'Total Error: {prediction_error["total_error"]:.16f}\n')
          f.write(f'Dynamics Error: {prediction_error["dynamics_error"]:.16f}\n')
          f.write(f'Reconstruction Error: {prediction_error["reconstruction_error"]:.16f}\n')
      else:
          f.write(f'Avg Prediction Error (unnormalized) {prediction_error:.16f}')

def plot_lipschitz_dist(lipschitz_coeff, folder_name, model_name=None, lipschitz_constraint=None):
    model_name = f"{model_name}_" if model_name else ""
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle("Local Lipschitz Coefficient")
    ax.hist(lipschitz_coeff, density=True, bins=50)
    if lipschitz_constraint:
        ax.axvline(lipschitz_constraint, linestyle="--", color="black")
    path = os.path.join(folder_name, f"{model_name}train_local_lipschitz.png")
    fig.savefig(path)

def exists_prev_output(output_folder, config):
    f1 = os.path.join(output_folder, "dynamics.pkl")
    f2 = os.path.join(output_folder, "statistics.txt")
    return os.path.exists(f1) and os.path.exists(f2)

def main():
    arg_parser = construct_parser()
    config = parse_config(arg_parser)
    output_folder = config.output.dynamics
    os.makedirs(output_folder, exist_ok=True)
    print('='*60)
    print(f"Config:")
    print('='*60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print('='*60)

    if exists_prev_output(output_folder, config) and not config.overwrite:
        print(f"Found existing results in {output_folder}, quit")
        exit(0)

    seed(config.seed)
    print('='*60)
    print(f"Loading data using flexible data loader")
    print('='*60)
 
    # Use flexible data loading system
    use_images = getattr(config.data, 'use_images', False)
    
    # Load data using flexible loader
    data_result = load_data_flexible(config.data)
    
    # Check if result is a Dataset (from H5pyLoader) or numpy arrays
    from torch.utils.data import Dataset
    from correct_il.datasets import H5pyImageDataset
    
    if isinstance(data_result, (Dataset, H5pyImageDataset)):
        # H5pyLoader returns a Dataset directly
        dataset = data_result
        use_images = True  # H5pyLoader is only for images
        image_shape = dataset.image_shape
        images = dataset
        next_images = dataset  # Will be accessed via dataset
        actions = dataset  # Will be accessed via dataset
        s = None
        sp = None
        print(f"Using H5pyImageDataset with {len(dataset)} samples")
        print(f"Image shape: {image_shape}, Action dim: {dataset.act_dim}")
    else:
        # Traditional loader returns numpy arrays
        obs, actions, next_obs = data_result
        
        # Determine if data is images or states
        # Check if observations are images (4D with spatial dimensions)
        if len(obs.shape) == 4 and obs.shape[1] > 10 and obs.shape[2] > 10:
            # Likely images: (N, H, W, C)
            use_images = True
            # Normalize images to [0, 1] range (from [0, 255])
            if obs.max() > 1.0:
                print(f"Normalizing images from [0, 255] to [0, 1]...")
                images = obs.astype(np.float32) / 255.0
                next_images = next_obs.astype(np.float32) / 255.0
            else:
                images = obs.astype(np.float32)
                next_images = next_obs.astype(np.float32)
            image_shape = (images.shape[1], images.shape[2], images.shape[3])  # (H, W, C)
            s = None
            sp = None
            print(f"Detected image data: shape {images.shape}, range [{images.min():.3f}, {images.max():.3f}]")
        elif use_images:
            # Explicitly requested images but might be flattened
            image_shape = getattr(config.data, 'image_shape', None)
            if image_shape is None:
                raise ValueError("use_images=True but observations are not 4D and image_shape not provided")
            H, W, C = image_shape
            # Reshape flattened observations to images
            if obs.shape[1] == H * W * C:
                images = obs.reshape(len(obs), H, W, C).astype(np.float32)
                next_images = next_obs.reshape(len(next_obs), H, W, C).astype(np.float32)
                # Normalize images to [0, 1] range (from [0, 255])
                if images.max() > 1.0:
                    print(f"Normalizing images from [0, 255] to [0, 1]...")
                    images = images / 255.0
                    next_images = next_images / 255.0
            else:
                raise ValueError(f"Observation dimension {obs.shape[1]} doesn't match image_shape {image_shape}")
            s = None
            sp = None
            print(f"Reshaped to image data: shape {images.shape}, range [{images.min():.3f}, {images.max():.3f}]")
        else:
            # State data
            s = obs
            sp = next_obs
            images = None
            next_images = None
            image_shape = None
            print(f"Detected state data: shape {s.shape}")

    print('='*60)
    print(f"Constructing dynamics model")
    print('='*60)
    # Construct Dynamics Model
    d_config = config.dynamics
    
    if use_images:
        # Use ImageWorldModel
        latent_dim = getattr(d_config, 'latent_dim', 100)
        reconstruction_weight = getattr(d_config, 'reconstruction_weight', 1.0)
        dynamics_weight = getattr(d_config, 'dynamics_weight', 1.0)
        
        # Get action dimension - handle both Dataset and numpy arrays
        if isinstance(actions, (Dataset, H5pyImageDataset)):
            act_dim = actions.act_dim
        else:
            act_dim = actions.shape[1]
        
        dynamics = ImageWorldModel(
            image_shape=image_shape,
            act_dim=act_dim,
            d_config=d_config,
            latent_dim=latent_dim,
            hidden_size=d_config.layers,
            fit_lr=d_config.lr,
            fit_wd=d_config.weight_decay,
            device="cpu" if config.no_gpu else "cuda",
            activation=d_config.activation,
            reconstruction_weight=reconstruction_weight,
            dynamics_weight=dynamics_weight
        )
        
        # Fit Dynamics Model
        print('='*60)
        print(f"Fitting image-based dynamics model...")
        print('='*60)
        
        # Check if wandb should be used
        use_wandb = getattr(config, 'use_wandb', False)
        wandb_project = getattr(config, 'wandb_project', None)
        wandb_name = getattr(config, 'wandb_name', None) or f"{config.env}_seed{config.seed}"
        
        # Check if reconstruction visualization should be enabled
        visualize_reconstructions = getattr(config, 'visualize_reconstructions', False)
        num_vis_samples = getattr(config, 'num_vis_samples', 4)
        
        # Set up visualization output directory
        if visualize_reconstructions:
            vis_output_dir = os.path.join(output_folder, "reconstruction_vis")
        else:
            vis_output_dir = None
        
        # Set up checkpoint directory
        checkpoint_dir = os.path.join(output_folder, "checkpoints")
        checkpoint_interval = getattr(config, 'checkpoint_interval', 10)  # Default: every 10 epochs
        visualize_interval = getattr(config, 'visualize_interval', 200)
        
        # Create dataset if we have numpy arrays, otherwise use the dataset directly
        if isinstance(images, (Dataset, H5pyImageDataset)):
            train_dataset = images
        else:
            from correct_il.datasets import ImageDynamicsDataset
            train_dataset = ImageDynamicsDataset(images, actions, next_images)
        
        train_loss = dynamics.fit_dynamics(
            train_dataset,
            train_epochs=d_config.train_epochs,
            batch_size=d_config.batch_size,
            set_transformations=True,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            visualize_reconstructions=visualize_reconstructions,
            num_vis_samples=num_vis_samples,
            vis_output_dir=vis_output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            visualize_interval=visualize_interval)
        
        # Report Validation Loss
        prediction_error = dynamics.eval_prediction_error(
            images, actions, next_images, d_config.batch_size)
        
        # Save distribution of local lipschitz coefficients over data
        local_L = dynamics.eval_lipschitz_coeff(images, actions, batch_size=1024)
    else:
        # Use regular WorldModel
        dynamics = WorldModel(s.shape[1], actions.shape[1], d_config=d_config,
                              hidden_size=d_config.layers,
                              fit_lr=d_config.lr,
                              fit_wd=d_config.weight_decay,
                              device="cpu" if config.no_gpu else "cuda",
                              activation=d_config.activation)
        
        # Fit Dynamics Model
        print('='*60)
        print(f"Fitting dynamics model...")
        print('='*60)
        train_loss = dynamics.fit_dynamics(
            s, actions, sp,
            batch_size=d_config.batch_size,
            train_epochs=d_config.train_epochs,
            set_transformations=True)
        
        # Report Validation Loss
        prediction_error = dynamics.eval_prediction_error(s, actions, sp, d_config.batch_size)
        
        # Save distribution of local lipschitz coefficients over data
        local_L = dynamics.eval_lipschitz_coeff(s, actions, batch_size=1024)

    # Save Model and config
    save_config_yaml(config, os.path.join(output_folder, "config.yaml"))

    fn = "dynamics_backward.pkl" if d_config.backward else "dynamics.pkl"
    with open(os.path.join(output_folder, "dynamics.pkl"), "wb") as f:
        pickle.dump(dynamics, f)

    # Save Training Loss
    save_loss(train_loss, output_folder, prediction_error, eval_loss=None)

    # Plot Lipschitz distribution
    if not use_images or local_L is not None:
        lipschitz_constraint = d_config.lipschitz_constraint**(1 + len(d_config.layers))
        plot_lipschitz_dist(local_L, output_folder, lipschitz_constraint=lipschitz_constraint)

if __name__ == "__main__":
    main()
