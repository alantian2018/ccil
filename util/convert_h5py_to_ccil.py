"""
Utility script to convert h5py demonstration data to CCIL format.

The input h5py file should have the structure:
    data/demo_X/actions
    data/demo_X/obs/agentview_image
    data/demo_X/obs/robot0_eye_in_hand_image

The output will be a pickle file containing a list of trajectories,
where each trajectory is a dict with "observations" and "actions" keys.
Observations are 2D arrays of shape (T, o) and actions are 2D arrays of shape (T, a).
"""

import argparse
import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def convert_h5py_to_ccil(h5py_path, output_path, flatten_images=True):
    """
    Convert h5py demonstration data to CCIL format.
    
    Args:
        h5py_path: Path to the input h5py file
        output_path: Path to save the output pickle file
        flatten_images: If True, flatten image arrays into 1D vectors. 
                       If False, keep images as 2D arrays (H, W, C) per timestep.
    """
    print(f"Loading h5py file from {h5py_path}")
    with h5py.File(h5py_path, 'r') as f:
        # Get all demo keys
        demo_keys = sorted([key for key in f['data'].keys() if key.startswith('demo_')])
        print(f"Found {len(demo_keys)} demonstrations")
        
        trajectories = []
        
        for demo_key in tqdm(demo_keys, desc="Processing demonstrations"):
            demo = f['data'][demo_key]
            
            # Extract actions
            if 'actions' not in demo:
                print(f"Warning: {demo_key} does not have 'actions', skipping")
                continue
            actions = np.array(demo['actions'])  # Shape: (T, a)
            
            # Extract observations
            if 'obs' not in demo:
                print(f"Warning: {demo_key} does not have 'obs', skipping")
                continue
            
            obs_group = demo['obs']
            
            # Extract images
            agentview_image = None
            eye_in_hand_image = None
            
            if 'agentview_image' in obs_group:
                agentview_image = np.array(obs_group['agentview_image'])  # Shape: (T, H, W, C)
            else:
                print(f"Warning: {demo_key} does not have 'obs/agentview_image', skipping")
                continue
            
            if 'robot0_eye_in_hand_image' in obs_group:
                eye_in_hand_image = np.array(obs_group['robot0_eye_in_hand_image'])  # Shape: (T, H, W, C)
            else:
                print(f"Warning: {demo_key} does not have 'obs/robot0_eye_in_hand_image', skipping")
                continue
            
            # Combine images into observations
            # Option 1: Flatten images and concatenate
            if flatten_images:
                # Flatten each image: (T, H, W, C) -> (T, H*W*C)
                agentview_flat = agentview_image.reshape(agentview_image.shape[0], -1)
                eye_in_hand_flat = eye_in_hand_image.reshape(eye_in_hand_image.shape[0], -1)
                # Concatenate: (T, H*W*C + H*W*C)
                observations = np.concatenate([agentview_flat, eye_in_hand_flat], axis=1)
            else:
                # Keep images as 2D arrays and stack along channel dimension
                # Shape: (T, H, W, 2*C) where 2*C = 6 (3 channels from each image)
                observations = np.concatenate([agentview_image, eye_in_hand_image], axis=3)
                # Flatten to (T, H*W*2*C) for CCIL format
                observations = observations.reshape(observations.shape[0], -1)
            
            # Ensure observations and actions have the same number of timesteps
            T_obs = observations.shape[0]
            T_act = actions.shape[0]
            
            if T_obs != T_act:
                print(f"Warning: {demo_key} has mismatched timesteps (obs: {T_obs}, actions: {T_act}), "
                      f"truncating to min({T_obs}, {T_act})")
                min_T = min(T_obs, T_act)
                observations = observations[:min_T]
                actions = actions[:min_T]
            
            # Create trajectory dict
            trajectory = {
                "observations": observations.astype(np.float32),
                "actions": actions.astype(np.float32)
            }
            
            trajectories.append(trajectory)
    
    print(f"\nConverted {len(trajectories)} trajectories")
    if len(trajectories) > 0:
        print(f"Observation shape: {trajectories[0]['observations'].shape}")
        print(f"Action shape: {trajectories[0]['actions'].shape}")
    
    # Save to pickle file
    print(f"\nSaving to {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert h5py demonstration data to CCIL format"
    )
    parser.add_argument(
        'h5py_path',
        type=str,
        help='Path to input h5py file'
    )
    parser.add_argument(
        'output_name',
        type=str,
       
        help='Path to output pickle file'
    )
 
    args = parser.parse_args()
    output_path = f"data/{args.output_name}.pkl"
    print('='*60)
    print(f"Output path: {output_path}")
    print(f"H5PY path: {args.h5py_path}")
    print(f"Flatten images: {args.flatten}")
    print('='*60)
    convert_h5py_to_ccil(
        args.h5py_path,
        output_path,
      
    )


if __name__ == '__main__':
    main()

