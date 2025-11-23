# Inspired by https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py
# Modified by Abhay Deshpande, Kay Ke, and Yunchu Zhang

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Callable, Sequence, Tuple, Optional
from functools import partial
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from correct_il.models.spectral_norm import apply_spectral_norm

class WorldModel:
    def __init__(self, state_dim, act_dim, d_config,
                 hidden_size=(64,64),
                 fit_lr=1e-3,
                 fit_wd=0.0,
                 device='cpu',
                 activation='relu'):
        self.state_dim, self.act_dim = state_dim, act_dim
        self.device = device if device != 'gpu' else 'cuda'

        # construct the dynamics model
        self.dynamics_net = DynamicsNet(
            state_dim, act_dim, hidden_size).to(self.device)
        self.dynamics_net.set_transformations()  # in case device is different from default, it will set transforms correctly
        if "spectral_normalization" in d_config.lipschitz_type:
            self.dynamics_net.enable_spectral_normalization(d_config.lipschitz_constraint)
        if activation == 'tanh':
            self.dynamics_net.nonlinearity = torch.tanh
        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_wd)

        if "slack" in d_config.lipschitz_type:
            self.learnable_lambda = DynamicsNet(state_dim, act_dim, hidden_size,out_dim=1).to(self.device)
            self.learnable_lambda.set_transformations()
            self.dynamics_opt = torch.optim.Adam(list(self.dynamics_net.parameters())+list(self.learnable_lambda.parameters()), lr=fit_lr, weight_decay=fit_wd)

        """loss_fn is of the signature (X, Y_pred, Y) -> {loss}"""
        self.loss_fn = construct_loss_fn(d_config, self) # default loss is MSE on Y

    def to(self, device):
        self.dynamics_net.to(device)
        self.device = device

    def is_cuda(self):
        return next(self.dynamics_net.parameters()).is_cuda

    def predict(self, s, a):
        s = torch.as_tensor(s).float().to(self.device)
        a = torch.as_tensor(a).float().to(self.device)
        s_next = s + self.dynamics_net.predict(s, a)
        return s_next

    def cal_lambda(self, s, a):
        s = torch.as_tensor(s).float().to(self.device)
        a = torch.as_tensor(a).float().to(self.device)
        return self.learnable_lambda.predict(s, a)

    def f(self, s, a):
        return self.dynamics_net.predict(s, a)

    def init_normalization_to_data(self, s, a, sp, set_transformations=True):
        # set network transformations
        if set_transformations:
            s_shift, a_shift = torch.mean(s, dim=0), torch.mean(a, dim=0)
            s_scale, a_scale = torch.mean(torch.abs(s - s_shift), dim=0), torch.mean(torch.abs(a - a_shift), dim=0)
            out_shift = torch.mean(sp-s, dim=0)
            out_scale = torch.mean(torch.abs(sp-s-out_shift), dim=0)
            self.dynamics_net.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)
        else:
            s_shift     = torch.zeros(self.state_dim).cuda()
            s_scale    = torch.ones(self.state_dim).cuda()
            a_shift     = torch.zeros(self.act_dim).cuda()
            a_scale    = torch.ones(self.act_dim).cuda()
            out_shift   = torch.zeros(self.state_dim).cuda()
            out_scale  = torch.ones(self.state_dim).cuda()
        return (s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)

    def fit_dynamics(self, s, a, sp, batch_size, train_epochs, max_steps=1e10,
                     set_transformations=True, *args, **kwargs):
        # move data to correct devices
        assert type(s) == type(a) == type(sp)
        assert s.shape[0] == a.shape[0] == sp.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            sp = torch.from_numpy(sp).float()

        s = s.to(self.device); a = a.to(self.device); sp = sp.to(self.device)
        s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = \
            self.init_normalization_to_data(s, a, sp, set_transformations)

        # prepare data for learning
        # note how Y is normalized & residual
        X = (s, a) ; Y = (sp - s - out_shift) / (out_scale + 1e-8)
        print('starting training...')
        return fit_model(
            self.dynamics_net, X, Y, self.dynamics_opt, self.loss_fn,
            batch_size, train_epochs, max_steps=max_steps)

    @torch.no_grad()
    def eval_lipschitz_coeff(self, s, a, batch_size=None):
        batch_size = batch_size if batch_size else len(s)
        def predict_concat(state_action):
            obs = state_action[...,:s.shape[-1]]
            act = state_action[...,s.shape[-1]:]
            return self.predict(obs, act)
        # compute batched jacobian with vectorization
        import functorch
        jac_fn = functorch.vmap(functorch.jacrev(predict_concat))

        s_a = np.concatenate([s, a], axis=-1)
        lipschits_coeffs = []
        for i in tqdm(range(0, len(s), batch_size)):
            batch = torch.as_tensor(s_a[i:i+batch_size], dtype=torch.float32, device=self.device)
            jacs = jac_fn(batch)
            assert jacs.shape == (batch.shape[0], s.shape[-1], batch.shape[-1])
            local_L = torch.linalg.norm(jacs, ord=2, dim=(-2,-1)).cpu().numpy()
            lipschits_coeffs.append(local_L)
        return np.concatenate(lipschits_coeffs, axis=0)

    @torch.no_grad()
    def eval_prediction_error(self, s, a, s_next, batch_size):
        s_next = torch.as_tensor(s_next).float().to(self.device)

        transforms = self.dynamics_net.get_params()['transforms']
        s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = transforms

        i = 0
        err_norms = []
        num_steps = int(s.shape[0] // batch_size)
        for _ in range(num_steps):
            sp = self.predict(s[i:i+batch_size], a[i:i+batch_size])
            error = (sp - s_next[i:i+batch_size]).norm(dim=-1)
            i += batch_size
            err_norms += error.cpu().tolist()
        return np.mean(err_norms)

def eval_prediction_error_images(images, actions, next_images, batch_size,
                                 image_encoder, image_decoder, dynamics_net, device,
                                 reconstruction_weight=1.0, dynamics_weight=1.0):
    """
    Evaluate prediction error for image-based dynamics model.
    
    Uses the full Dreamer-style loss (both components):
    - Dynamics loss: MSE in latent space (on normalized residuals)
    - Reconstruction loss: MSE in image space (pixel-wise)
    
    This matches the training loss, providing consistent evaluation.
    
    Args:
        images: (N, H, W, C) or (N, C, H, W) current images
        actions: (N, act_dim) actions
        next_images: (N, H, W, C) or (N, C, H, W) ground truth next images
        batch_size: Batch size for evaluation
        image_encoder: ImageEncoder instance
        image_decoder: ImageDecoder instance
        dynamics_net: DynamicsNet instance
        device: Device to run on
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
        dynamics_weight: Weight for dynamics loss (default: 1.0)
    
    Returns:
        Dictionary with:
        - 'total_error': Combined error (dynamics_weight * dynamics_loss + reconstruction_weight * reconstruction_loss)
        - 'dynamics_error': Mean dynamics loss in latent space
        - 'reconstruction_error': Mean reconstruction loss in image space
    """
    images = torch.as_tensor(images).float().to(device)
    actions = torch.as_tensor(actions).float().to(device)
    next_images = torch.as_tensor(next_images).float().to(device)
    
    # Get normalization parameters
    transforms = dynamics_net.get_params()['transforms']
    s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = transforms
    
    i = 0
    dynamics_errors = []
    reconstruction_errors = []
    num_steps = int(images.shape[0] // batch_size)
    
    with torch.no_grad():
        for _ in range(num_steps):
            batch_images = images[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            batch_next_images = next_images[i:i+batch_size]
            
            # Encode images to latent states
            s_latent = image_encoder(batch_images)
            sp_latent_true = image_encoder(batch_next_images)
            
            # Normalize inputs for dynamics net
            s_latent_norm = (s_latent - s_shift) / (s_scale + 1e-8)
            a_norm = (batch_actions - a_shift) / (a_scale + 1e-8)
            
            # Compute true normalized residual
            residual_true = sp_latent_true - s_latent
            residual_true_norm = (residual_true - out_shift) / (out_scale + 1e-8)
            
            # Predict normalized residual using dynamics net
            residual_pred_norm = dynamics_net.forward(s_latent_norm, a_norm)
            
            # 1. Dynamics loss: MSE in normalized latent space
            dynamics_loss = F.mse_loss(residual_pred_norm, residual_true_norm, reduction='none')
            dynamics_loss = dynamics_loss.mean(dim=1)  # Average over latent dimensions
            dynamics_errors += dynamics_loss.cpu().tolist()
            
            # 2. Reconstruction loss: decode and compare images
            # Denormalize predicted residual
            residual_pred = residual_pred_norm * (out_scale + 1e-8) + out_shift
            sp_latent_pred = s_latent + residual_pred
            
            # Decode predicted next images
            next_images_pred = image_decoder(sp_latent_pred)
            
            # Handle image format conversion
            if batch_next_images.dim() == 4:
                if batch_next_images.shape[-1] in [1, 3]:  # (B, H, W, C) format
                    if next_images_pred.shape[1] in [1, 3]:
                        next_images_pred = next_images_pred.permute(0, 2, 3, 1)
                else:  # (B, C, H, W) format
                    if next_images_pred.shape[-1] in [1, 3]:
                        next_images_pred = next_images_pred.permute(0, 3, 1, 2)
            

            # Reconstruction loss: pixel-wise MSE
            reconstruction_loss = F.mse_loss(next_images_pred, batch_next_images, reduction='none')
         
            reconstruction_loss = reconstruction_loss.flatten(start_dim=1).mean(dim=1)  # Average over pixels
            reconstruction_errors += reconstruction_loss.cpu().tolist()
    
    # Compute mean errors
    mean_dynamics_error = np.mean(dynamics_errors)
    mean_reconstruction_error = np.mean(reconstruction_errors)
    total_error = dynamics_weight * mean_dynamics_error + reconstruction_weight * mean_reconstruction_error
    
    return {
        'total_error': total_error,
        'dynamics_error': mean_dynamics_error,
        'reconstruction_error': mean_reconstruction_error
    }

    # NOTE: For image-based models, eval_prediction_error would differ as follows:
    # 
    # @torch.no_grad()
    # def eval_prediction_error_images(self, images, actions, next_images, batch_size):
    #     """
    #     Evaluate prediction error for image-based dynamics model.
    #     
    #     Key differences from state-based eval_prediction_error:
    #     1. Takes images as input instead of states (s, s_next)
    #     2. Encodes images to latent states before prediction
    #     3. Can compute error in multiple ways:
    #        - Pixel-wise MSE in image space
    #        - MSE in latent space (faster, but less interpretable)
    #        - Perceptual loss (using pre-trained features)
    #     4. Error is typically computed in image space for interpretability
    #     
    #     Args:
    #         images: (N, H, W, C) current images
    #         actions: (N, act_dim) actions
    #         next_images: (N, H, W, C) ground truth next images
    #         batch_size: Batch size for evaluation
    #     
    #     Returns:
    #         Mean pixel-wise MSE error
    #     """
    #     next_images = torch.as_tensor(next_images).float().to(self.device)
    #     
    #     i = 0
    #     err_norms = []
    #     num_steps = int(images.shape[0] // batch_size)
    #     for _ in range(num_steps):
    #         # Predict next images
    #         next_images_pred = self.predict(images[i:i+batch_size], actions[i:i+batch_size])
    #         
    #         # Compute pixel-wise MSE error
    #         # Option 1: Direct pixel MSE
    #         error = F.mse_loss(next_images_pred, next_images[i:i+batch_size], reduction='none')
    #         error = error.view(error.shape[0], -1).mean(dim=1)  # Average over pixels
    #         
    #         # Option 2: L2 norm per image
    #         # error = (next_images_pred - next_images[i:i+batch_size]).norm(dim=(1,2,3))
    #         
    #         # Option 3: Latent space error (faster but less interpretable)
    #         # s_latent = self.encode_images(images[i:i+batch_size])
    #         # sp_latent_pred = self.encode_images(next_images_pred)
    #         # sp_latent_true = self.encode_images(next_images[i:i+batch_size])
    #         # error = (sp_latent_pred - sp_latent_true).norm(dim=-1)
    #         
    #         i += batch_size
    #         err_norms += error.cpu().tolist()
    #     return np.mean(err_norms)

def construct_loss_fn(d_config, dynamics):
    if d_config.lipschitz_type == "soft_sampling":
      return partial(soft_sampling_lipschitz_loss,
                                lipschitz_constraint=d_config.lipschitz_constraint,
                                sampling_eps=d_config.soft_lipschitz_sampling_eps,
                                soft_lipschitz_penalty_weight=d_config.soft_lipschitz_penalty_weight,
                                n_samples=d_config.soft_lipschitz_n_samples,
                                predict_fn=dynamics.predict)
    elif d_config.lipschitz_type == "soft_sampling_slack":
      return partial(soft_sampling_lipschitz_slack_loss,
                                lipschitz_constraint=d_config.lipschitz_constraint,
                                sampling_eps=d_config.soft_lipschitz_sampling_eps,
                                soft_lipschitz_penalty_weight=d_config.soft_lipschitz_penalty_weight,
                                n_samples=d_config.soft_lipschitz_n_samples,
                                predict_fn=dynamics.predict,
                                lambda_fn=dynamics.cal_lambda)
    elif d_config.lipschitz_type == "slack" or d_config.lipschitz_type == "spectral_normalization_slack":
      slack_weight = float(d_config.slack_weight) if 'slack_weight' in d_config else 1.0
      return partial(slack_loss, lambda_fn=dynamics.cal_lambda, slack_weight=slack_weight)
    elif d_config.lipschitz_type == "soft_jac":
      return partial(soft_lipschitz_loss,
                                lipschitz_constraint=d_config.lipschitz_constraint,
                                soft_lipschitz_penalty_weight=d_config.soft_lipschitz_penalty_weight,
                                predict_fn=dynamics.predict)
    else:
      return wrapper_mse_loss

def wrapper_mse_loss(_, Y_pred, Y):
    mse_loss = F.mse_loss(Y_pred, Y)
    return {
        'loss': mse_loss,
        'mean_error': F.mse_loss(Y_pred, Y).detach().cpu().numpy()
    }

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using ResNet features.
    Computes L2 distance between feature maps from a pre-trained ResNet network.
    """
    def __init__(self, resnet_type='resnet18', device='cuda'):
        """
        Args:
            resnet_type: Type of ResNet to use ('resnet18' or 'resnet34')
            device: Device to run on
        """
        super(PerceptualLoss, self).__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for perceptual loss. Install with: pip install torchvision")
        
        # Load pre-trained ResNet
        # Handle both old and new torchvision API
        try:
            # New API (torchvision 0.13+)
            if resnet_type == 'resnet18':
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
            elif resnet_type == 'resnet34':
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
            else:
                raise ValueError(f"Unsupported ResNet type: {resnet_type}. Use 'resnet18' or 'resnet34'")
        except (AttributeError, TypeError):
            # Old API (torchvision < 0.13)
            if resnet_type == 'resnet18':
                resnet = models.resnet18(pretrained=True).to(device)
            elif resnet_type == 'resnet34':
                resnet = models.resnet34(pretrained=True).to(device)
            else:
                raise ValueError(f"Unsupported ResNet type: {resnet_type}. Use 'resnet18' or 'resnet34'")
        
        resnet.eval()
        
        # Freeze ResNet parameters
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Extract feature layers from ResNet
        # We'll extract features from: conv1+bn1+relu, layer1, layer2, layer3, layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Normalization for ImageNet pre-trained models
        # ResNet expects input in range [0, 1] with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def preprocess_image(self, img):
        """
        Preprocess image for ResNet network.
        Converts from [0, 1] range to ImageNet normalized range.
        Expects input in (B, H, W, C) or (B, C, H, W) format.
        Handles different channel counts by converting to 3-channel RGB.
        """
        # Convert to (B, C, H, W) format if needed
        if img.dim() == 4:
            if img.shape[-1] in [1, 3]:  # (B, H, W, C) format
                img = img.permute(0, 3, 1, 2)
        
        # Handle channel count: ResNet expects 3 channels
        if img.shape[1] == 1:
            # Grayscale: replicate to 3 channels
            img = img.repeat(1, 3, 1, 1)
        elif img.shape[1] > 3:
            # More than 3 channels: take first 3
            img = img[:, :3, :, :]
        # If 3 channels, use as is
        
        # Normalize to ImageNet stats
        img = (img - self.mean) / self.std
        return img
    
    def extract_features(self, x):
        """Extract features from multiple layers of ResNet."""
        x = self.preprocess_image(x)
        features = []
        
        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # After conv1
        
        # Maxpool
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        features.append(x)  # After layer1
        
        x = self.layer2(x)
        features.append(x)  # After layer2
        
        x = self.layer3(x)
        features.append(x)  # After layer3
        
        x = self.layer4(x)
        features.append(x)  # After layer4
        
        return features
    
    def forward(self, pred, target):
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted images (B, H, W, C) or (B, C, H, W)
            target: Target images (B, H, W, C) or (B, C, H, W)
        
        Returns:
            Perceptual loss scalar
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        # Compute L2 loss for each feature layer and average
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)

def dreamer_image_loss(X, Y_pred, Y, image_encoder, image_decoder, 
                       next_images_true, out_shift, out_scale,
                       reconstruction_weight=1.0, dynamics_weight=1.0,
                       dynamics_net=None, lipschitz_constraint=None,
                       soft_lipschitz_penalty_weight=None, 
                       soft_lipschitz_sampling_eps=None,
                       soft_lipschitz_n_samples=None,
                       reconstruction_weights=None,
                       use_perceptual_loss=False,
                       mse_weight=0.85,
                       perceptual_weight=0.15,
                       perceptual_loss_fn=None):
    """
    Dreamer-style loss for image-based dynamics models.
    
    Combines:
    1. Reconstruction loss: MSE between predicted and true images (in image space)
    2. Dynamics loss: MSE in latent space (on normalized residuals)
    3. (Optional) Lipschitz penalty: Applied to latent space dynamics
    
    Similar to DreamerV2/V3 which uses:
    - Image reconstruction loss (MSE or cross-entropy)
    - Latent dynamics loss (MSE in latent space)
    
    Args:
        X: Tuple of (s_latent, actions) - current latent states and actions
        Y_pred: Predicted latent state residuals (normalized)
        Y: True latent state residuals (normalized)
        image_encoder: ImageEncoder instance
        image_decoder: ImageDecoder instance
        next_images_true: Ground truth next images (B, H, W, C) or (B, C, H, W)
        out_shift: Output shift for denormalization
        out_scale: Output scale for denormalization
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
        dynamics_weight: Weight for dynamics loss (default: 1.0)
        dynamics_net: DynamicsNet instance (for Lipschitz penalty)
        lipschitz_constraint: Lipschitz constraint value (if None, no penalty)
        soft_lipschitz_penalty_weight: Weight for Lipschitz penalty
        soft_lipschitz_sampling_eps: Epsilon for Lipschitz sampling
        soft_lipschitz_n_samples: Number of samples for Lipschitz estimation
        reconstruction_weights: Optional weight matrix of same shape as next_images_pred/next_images_true
                               for weighted MSE reconstruction loss (default: None)
    
    Returns:
        Dictionary with loss components
    """
    s_latent, actions = X
    
    # 1. Dynamics loss in latent space (MSE on normalized residuals)
    # This is the standard dynamics prediction loss
    dynamics_loss = F.mse_loss(Y_pred, Y)
    
    # 2. Reconstruction loss: decode predicted next images and compare with ground truth
    # Denormalize predicted residual to get actual residual
    residual_pred = Y_pred * (out_scale + 1e-8) + out_shift
    
    # Compute predicted next latent state
    sp_latent_pred = s_latent + residual_pred
    
    # Decode predicted next images
    next_images_pred = image_decoder(sp_latent_pred)
    
    # Handle image format conversion
    # Ensure next_images_pred matches next_images_true format
    if next_images_true.dim() == 4:
        if next_images_true.shape[-1] in [1, 3]:  # (B, H, W, C) format
            # next_images_pred is (B, C, H, W), convert to (B, H, W, C)
            if next_images_pred.shape[1] in [1, 3]:
                next_images_pred = next_images_pred.permute(0, 2, 3, 1)
        else:  # (B, C, H, W) format
            # next_images_pred is (B, C, H, W), keep as is
            if next_images_pred.shape[-1] in [1, 3]:
                next_images_pred = next_images_pred.permute(0, 3, 1, 2)
    
    # Images should already be normalized to [0, 1] outside the model
    
    # Compute MSE reconstruction loss with optional weights
    mse_reconstruction_loss = None
    if reconstruction_weights is not None:
        # Ensure weights are on the same device and have the same shape
        reconstruction_weights = reconstruction_weights.to(next_images_pred.device)
        if reconstruction_weights.shape != next_images_pred.shape:
            raise ValueError(f"reconstruction_weights shape {reconstruction_weights.shape} must match "
                           f"next_images_pred shape {next_images_pred.shape}")
        # Weighted MSE: loss = (weights * (pred - target)**2).sum() / weights.sum()
        squared_error = (next_images_pred - next_images_true) ** 2
        # weighted sum per sample: [B]
        num = (reconstruction_weights * squared_error).sum(dim=(1,2,3))

        # weight mass per sample: [B]
        den = reconstruction_weights.sum(dim=(1,2,3)) + 1e-8

        # mean over batch
        mse_reconstruction_loss = (num / den).mean()
    else:
        mse_reconstruction_loss = F.mse_loss(next_images_pred, next_images_true)
    
    # Compute perceptual loss if enabled
    perceptual_reconstruction_loss = None
    if use_perceptual_loss and perceptual_loss_fn is not None:
        perceptual_reconstruction_loss = perceptual_loss_fn(next_images_pred, next_images_true)
    
    # Combine MSE and perceptual losses based on configuration
    if use_perceptual_loss and perceptual_loss_fn is not None:
        # Weighted combination: mse_weight * MSE + perceptual_weight * Perceptual
        reconstruction_loss = mse_weight * mse_reconstruction_loss + perceptual_weight * perceptual_reconstruction_loss
    else:
        # Use only MSE loss
        reconstruction_loss = mse_reconstruction_loss
    
    # 3. Lipschitz penalty (if enabled) - applied to latent space dynamics
    lipschitz_penalty = None
    if (lipschitz_constraint is not None and 
        soft_lipschitz_penalty_weight is not None and 
        soft_lipschitz_sampling_eps is not None and
        soft_lipschitz_n_samples is not None and
        dynamics_net is not None):
        
        # Create predict function that works on latent states
        # This function takes (s_latent, action) and returns next latent state
        def predict_from_latent(s_lat, a_lat):
            """Predict next latent state from latent state and action."""
            # dynamics_net.predict returns the residual, so add it to s_lat
            return s_lat + dynamics_net.predict(s_lat, a_lat)
        
        # Estimate Lipschitz constant using sampling
        # sample_estimate_lipschitz expects: (s, a, Y_pred, predict_fn, n_samples, eps)
        # where Y_pred is the predicted output (next latent state in our case)
        noise_mag, output_diff = sample_estimate_lipschitz(
            s_latent, actions, sp_latent_pred, predict_from_latent, 
            soft_lipschitz_n_samples, soft_lipschitz_sampling_eps
        )
        lipschitz_penalty = torch.relu(output_diff - lipschitz_constraint * noise_mag)
        lipschitz_penalty = soft_lipschitz_penalty_weight * lipschitz_penalty.mean()
    
    # Combined loss (Dreamer-style: weighted sum of dynamics and reconstruction)
    # Lipschitz penalty is part of dynamics loss (constrains the dynamics function)
    dynamics_loss_with_penalty = dynamics_loss
    if lipschitz_penalty is not None:
        dynamics_loss_with_penalty = dynamics_loss + lipschitz_penalty
    
    total_loss = dynamics_weight * dynamics_loss_with_penalty + reconstruction_weight * reconstruction_loss
    
    result = {
        'loss': total_loss,
        'dynamics_loss': dynamics_loss.detach().cpu().numpy(),
        'reconstruction_loss': reconstruction_loss.detach().cpu().numpy(),
        'mse_reconstruction_loss': mse_reconstruction_loss.detach().cpu().numpy(),
        'mean_error': F.mse_loss(Y_pred, Y).detach().cpu().numpy()
    }
    
    if perceptual_reconstruction_loss is not None:
        result['perceptual_reconstruction_loss'] = perceptual_reconstruction_loss.detach().cpu().numpy()
    
    if lipschitz_penalty is not None:
        result['lipschitz_penalty'] = lipschitz_penalty.detach().cpu().numpy()
    
    return result

def sample_estimate_lipschitz(s, a, Y_pred, predict_fn, n_samples, sampling_eps):
  # repeat the rows so we can perturb each element multiple times
  s = torch.as_tensor(s).to(Y_pred.device).repeat_interleave(n_samples, dim=0)
  a = torch.as_tensor(a).to(Y_pred.device).repeat_interleave(n_samples, dim=0)
  Y_pred_repeat = Y_pred.repeat_interleave(n_samples, dim=0)
  s_noise = torch.randn_like(s, device=s.device)
  a_noise = torch.randn_like(a, device=a.device)
  noisy_s = s + s_noise * sampling_eps
  noisy_a = a + a_noise * sampling_eps
  noisy_output = predict_fn(noisy_s, noisy_a)
  noise_mag = torch.linalg.norm(torch.cat([s_noise, a_noise], dim=-1), dim=-1) # batch_size * n_samples
  output_diff = torch.linalg.norm(Y_pred_repeat - noisy_output, dim=-1)
  return noise_mag, output_diff

def soft_sampling_lipschitz_loss(
            X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
            lipschitz_constraint: float, sampling_eps: float, soft_lipschitz_penalty_weight: float, n_samples: int,
            predict_fn: Callable[[torch.Tensor,torch.Tensor], torch.Tensor]):
  s, a = X
  noise_mag, output_diff = sample_estimate_lipschitz(s, a, Y_pred, predict_fn, n_samples, sampling_eps)
  lipschitz_penalty = torch.relu(output_diff - lipschitz_constraint * noise_mag)
  lipschitz_penalty = soft_lipschitz_penalty_weight * lipschitz_penalty.mean()
  mse_loss = F.mse_loss(Y_pred, Y)
  return {
      'loss': mse_loss + lipschitz_penalty,
      'mse_loss_tensor' : mse_loss,
      'mse_loss' : mse_loss.detach().to('cpu').numpy(),
      'lipschitz_penalty': lipschitz_penalty.detach().to('cpu').numpy()
  }

def soft_sampling_lipschitz_slack_loss(
            X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
            lipschitz_constraint: float, sampling_eps: float, soft_lipschitz_penalty_weight: float, n_samples: int,
            predict_fn: Callable[[torch.Tensor,torch.Tensor], torch.Tensor], lambda_fn):
  s, a = X
  noise_mag, output_diff = sample_estimate_lipschitz(s, a, Y_pred, predict_fn, n_samples, sampling_eps)
  lipschitz_penalty = torch.relu(output_diff - lipschitz_constraint * noise_mag)
  slack = lambda_fn(s, a)
  lipschitz_penalty = slack * lipschitz_penalty + 1 - torch.exp(- 0.1 * torch.abs(slack).mean())
  lipschitz_penalty = soft_lipschitz_penalty_weight * lipschitz_penalty.mean()
  mse_loss = F.mse_loss(Y_pred, Y)
  return {
      'loss': mse_loss + lipschitz_penalty,
      'mse_loss_tensor' : mse_loss,
      'mse_loss' : mse_loss.detach().to('cpu').numpy(),
      'lipschitz_penalty': lipschitz_penalty.detach().to('cpu').numpy()
  }

def slack_loss(X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
               lambda_fn=None, slack_weight=0.1):
  s,a = X
  slack = lambda_fn(s, a)
  slack = torch.nn.Sigmoid()(slack) # squeeze to 0~1
  mse_loss = F.mse_loss(Y_pred, Y, reduction='none')
  weighted_mse_loss = torch.linalg.norm(slack * mse_loss, dim=-1).mean()
  avg_slack_size = slack.mean()

  return {
      'loss': weighted_mse_loss.mean() - slack_weight * avg_slack_size,
      'mse_loss_tensor' : mse_loss,
      'mse_loss' : mse_loss.norm(dim=-1).mean().detach().cpu().numpy(),
      'weighted_mse_loss' : weighted_mse_loss.mean().detach().to('cpu').numpy(),
      'slack_penalty': avg_slack_size.detach().to('cpu').numpy(),
      'slack_std': torch.std(slack).detach().cpu().numpy()
  }

def soft_lipschitz_loss(X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
                        lipschitz_constraint: float, soft_lipschitz_penalty_weight: float,
                        predict_fn: Callable[[torch.Tensor,torch.Tensor], torch.Tensor]):
  s, a = X
  s = torch.as_tensor(s).to(Y_pred.device)
  a = torch.as_tensor(a).to(Y_pred.device)
  def predict_concat(state_action):
    obs = state_action[...,:s.shape[-1]]
    act = state_action[...,s.shape[-1]:]
    return predict_fn(obs, act)
  # compute batched jacobian with vectorization
  import functorch
  jacs = functorch.vmap(functorch.jacrev(predict_concat))(torch.cat([s, a], dim=-1))
  assert jacs.shape == (s.shape[0], s.shape[-1], s.shape[-1]+a.shape[-1])
  local_L = torch.linalg.norm(jacs, ord=2, dim=(-2,-1)) # local lipschitz coeff is spectral norm of jacobian
  lipschitz_penalty = torch.relu(local_L - lipschitz_constraint)
  lipschitz_penalty = soft_lipschitz_penalty_weight * lipschitz_penalty.mean()
  mse_loss = F.mse_loss(Y_pred, Y)
  return {
      'loss': mse_loss + lipschitz_penalty,
      'mse_loss' : mse_loss.detach().to('cpu').numpy(),
      'lipschitz_penalty': lipschitz_penalty.detach().to('cpu').numpy()
  }


def _apply_spectral_normalization_recursively(
    model: nn.Module,
    lipschitz_constraint: float) -> None:
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for m in module:
                _apply_spectral_normalization_recursively(m, lipschitz_constraint)
        else:
            if "weight" in module._parameters:
                apply_spectral_norm(module, lipschitz_constraint=lipschitz_constraint)


class ImageEncoder(nn.Module):
    """
    CNN-based image encoder that converts images to latent state vectors.

    """
    def __init__(self, image_shape=(256, 256, 3), latent_dim=512, 
                 image_mean=None, image_std=None):
        """
        Args:
            image_shape: (H, W, C) shape of input images
            latent_dim: Dimension of output latent state
            image_mean: Optional mean for normalization (if None, uses [0,1] normalization)
            image_std: Optional std for normalization
        """
        super(ImageEncoder, self).__init__()
        H, W, C = image_shape
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.image_mean = image_mean
        self.image_std = image_std
        
        # CNN encoder: (B, H, W, C) -> (B, latent_dim)
        # Using standard CNN architecture
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),  # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
        ])
        
        # Calculate flattened size after convolutions
        # After 4 stride-2 convs: 256 -> 128 -> 64 -> 32 -> 16
        conv_output_size = 256 * 16 * 16  # channels * H * W
        self.fc = nn.Linear(conv_output_size, latent_dim)
        
    
    def forward(self, images):
        """
        Args:
            images: (B, H, W, C) or (B, C, H, W) tensor
        Returns:
            latent: (B, latent_dim) tensor
        """
        assert images.max() <= 1.0 and images.min() >= 0.0, "Images should be normalized to [0, 1] range"
        # Handle both (B, H, W, C) and (B, C, H, W) formats
        # Check if last dimension is small (likely channels) vs if second dimension is small (likely channels first)
        if images.dim() == 4:
            # If last dim is small (channels: 1, 3, 6, etc.) and second dim is large (height), it's (B, H, W, C)
            # If second dim is small (channels) and last dim is large (width), it's (B, C, H, W)
            last_dim = images.shape[-1]
            second_dim = images.shape[1]
            # Typical image: (B, 256, 256, 6) -> last_dim=6 (small), second_dim=256 (large) -> needs permute
            # Already correct: (B, 6, 256, 256) -> second_dim=6 (small), last_dim=256 (large) -> no permute
            if last_dim <= 10 and second_dim > 10:
                # Assume (B, H, W, C) -> convert to (B, C, H, W)
                images = images.permute(0, 3, 1, 2)
            # Otherwise assume it's already (B, C, H, W)


        # Apply CNN layers
        x = images
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
            else:
                x = layer(x)
        
        # Flatten and project to latent space
        x = x.flatten(start_dim=1)
        
        latent = self.fc(x)
        
        return latent


class ImageDecoder(nn.Module):
    """
    CNN-based image decoder that converts latent state vectors back to images.
    Outputs images in [0, 1] range (to match encoder normalization).
    To convert to [0, 255]: output * 255
    """
    def __init__(self, image_shape=(256, 256, 3), latent_dim=100):
        """
        Args:
            image_shape: (H, W, C) shape of output images
            latent_dim: Dimension of input latent state
        """
        super(ImageDecoder, self).__init__()
        H, W, C = image_shape
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        
        # Calculate flattened size for initial feature map
        # We'll start from 16x16 and upsample to 256x256
        initial_size = 256 * 16 * 16  # channels * H * W
        
        self.fc = nn.Linear(latent_dim, initial_size)
        
        # Transposed CNN decoder: (B, latent_dim) -> (B, H, W, C)
        self.deconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),  # 128x128 -> 256x256
            nn.Sigmoid()  # Output in [0, 1] range
        ])
        
    def forward(self, latent):
        """
        Args:
            latent: (B, latent_dim) tensor
        Returns:
            images: (B, C, H, W) tensor in [0, 1] range
        """
        # Project to feature map
        x = self.fc(latent)
        x = x.view(x.size(0), 256, 16, 16)  # Reshape to (B, C, H, W)
        
        # Apply transposed convolutions
        for layer in self.deconv_layers:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
            else:
                x = layer(x)
        # permute back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x


class ImageWorldModel:
    """
    Complete image-based world model that integrates:
    - ImageEncoder: images -> latent states
    - DynamicsNet: latent states + actions -> next latent states (residual)
    - ImageDecoder: latent states -> images
    
    This class provides the same interface as WorldModel but works with images.
    """
    def __init__(self, image_shape, act_dim, d_config,
                 latent_dim=100,
                 hidden_size=(64,64),
                 fit_lr=1e-4,
                 fit_wd=0.0,
                 device='cpu',
                 activation='relu',
                 reconstruction_weight=1.0,
                 dynamics_weight=1.0,
                 image_mean=None,
                 image_std=None,
                 image_shape_default=(256, 256, 3)):
        """
        Args:
            image_shape: (H, W, C) shape of input images (required, no default)
            act_dim: Action dimension
            latent_dim: Dimension of latent state after encoding
            d_config: Dynamics configuration (same as WorldModel)
            reconstruction_weight: Weight for reconstruction loss in Dreamer loss
            dynamics_weight: Weight for dynamics loss in Dreamer loss
            image_mean: Optional mean for image normalization
            image_std: Optional std for image normalization
            image_shape_default: Default image shape if image_shape is None (for backward compat)
        """
        if image_shape is None:
            image_shape = image_shape_default
        self.image_shape = image_shape
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.device = device if device != 'gpu' else 'cuda'
        self.reconstruction_weight = reconstruction_weight
        self.dynamics_weight = dynamics_weight
        
        # Image encoder: images -> latent states
        self.image_encoder = ImageEncoder(
            image_shape=image_shape,
            latent_dim=latent_dim,
            image_mean=image_mean,
            image_std=image_std
        ).to(self.device)
        
        # Image decoder: latent states -> images
        self.image_decoder = ImageDecoder(
            image_shape=image_shape,
            latent_dim=latent_dim
        ).to(self.device)
        
        # Dynamics net operates on latent states (not raw images)
        self.dynamics_net = DynamicsNet(
            state_dim=latent_dim,  # Works on encoded latent states
            act_dim=act_dim,
            hidden_size=hidden_size
        ).to(self.device)
        self.dynamics_net.set_transformations()
        
        if "spectral_normalization" in d_config.lipschitz_type:
            self.dynamics_net.enable_spectral_normalization(d_config.lipschitz_constraint)
        if activation == 'tanh':
            self.dynamics_net.nonlinearity = torch.tanh
            
        # Optimizer for all components
        params = list(self.image_encoder.parameters()) + \
                 list(self.image_decoder.parameters()) + \
                 list(self.dynamics_net.parameters())
        self.dynamics_opt = torch.optim.Adam(params, lr=fit_lr, weight_decay=fit_wd)
        
        if "slack" in d_config.lipschitz_type:
            self.learnable_lambda = DynamicsNet(
                latent_dim, act_dim, hidden_size, out_dim=1
            ).to(self.device)
            self.learnable_lambda.set_transformations()
            params = list(self.image_encoder.parameters()) + \
                     list(self.image_decoder.parameters()) + \
                     list(self.dynamics_net.parameters()) + \
                     list(self.learnable_lambda.parameters())
            self.dynamics_opt = torch.optim.Adam(params, lr=fit_lr, weight_decay=fit_wd)
        
        # Store d_config for loss function construction
        self.d_config = d_config
        # Note: loss_fn will be constructed in fit_dynamics using dreamer_image_loss
    
    def encode_images(self, images):
        """Encode images to latent states."""
        images = torch.as_tensor(images).float().to(self.device)
        return self.image_encoder(images)
    
    def decode_images(self, latent_states):
        """Decode latent states to images."""
        latent_states = torch.as_tensor(latent_states).float().to(self.device)
        images = self.image_decoder(latent_states)
        # Convert from (B, C, H, W) to (B, H, W, C) for consistency
        if images.dim() == 4:
            images = images.permute(0, 2, 3, 1)
        return images
    
    def predict(self, images, actions):
        """
        Predict next images given current images and actions.
        
        Args:
            images: (B, H, W, C) or (B, C, H, W) tensor
            actions: (B, act_dim) tensor
        
        Returns:
            next_images: (B, H, W, C) tensor
        """
        # Encode images to latent states
        s_latent = self.encode_images(images)
        a = torch.as_tensor(actions).float().to(self.device)
        
        # Predict next latent state (residual)
        s_next_latent = s_latent + self.dynamics_net.predict(s_latent, a)
        
        # Decode back to images
        next_images = self.decode_images(s_next_latent)
        return next_images
    
    def init_normalization_to_data(self, images, actions, next_images, set_transformations=True):
        """
        Initialize normalization statistics from image data.
        
        Args:
            images: (N, H, W, C) or (N, C, H, W) tensor of current images
            actions: (N, act_dim) tensor of actions
            next_images: (N, H, W, C) or (N, C, H, W) tensor of next images
            set_transformations: Whether to set normalization transformations
        
        Returns:
            Normalization parameters (same format as WorldModel)
        """
        # Encode images to latent states
        with torch.no_grad():
            s_latent = self.encode_images(images)
            sp_latent = self.encode_images(next_images)
            a = torch.as_tensor(actions).float().to(self.device)
        
        # Now normalize latent states (same as regular WorldModel)
        if set_transformations:
            s_shift, a_shift = torch.mean(s_latent, dim=0), torch.mean(a, dim=0)
            s_scale, a_scale = torch.mean(torch.abs(s_latent - s_shift), dim=0), \
                              torch.mean(torch.abs(a - a_shift), dim=0)
            out_shift = torch.mean(sp_latent - s_latent, dim=0)
            out_scale = torch.mean(torch.abs(sp_latent - s_latent - out_shift), dim=0)
            self.dynamics_net.set_transformations(s_shift, s_scale, a_shift, a_scale, 
                                                  out_shift, out_scale)
        else:
            s_shift = torch.zeros(self.latent_dim).to(self.device)
            s_scale = torch.ones(self.latent_dim).to(self.device)
            a_shift = torch.zeros(self.act_dim).to(self.device)
            a_scale = torch.ones(self.act_dim).to(self.device)
            out_shift = torch.zeros(self.latent_dim).to(self.device)
            out_scale = torch.ones(self.latent_dim).to(self.device)
        
        return (s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)
    
    def fit_dynamics(self, dataset, train_epochs, batch_size=None,
                     max_steps=1e10, set_transformations=True, use_wandb=False, 
                     wandb_project=None, wandb_name=None, visualize_reconstructions=False,
                     num_vis_samples=4, vis_output_dir=None, checkpoint_dir=None, 
                     checkpoint_interval=10, visualize_interval=200, reconstruction_weights=None, 
                     *args, **kwargs):
        """
        Train the image-based dynamics model using Dreamer-style loss.
        
        Args:
            dataset: Dataset object (ImageDynamicsDataset, H5pyImageDataset, etc.)
            train_epochs: Number of training epochs
            batch_size: Batch size for training (default: 32)
        """
        from correct_il.datasets import H5pyImageDataset
        from torch.utils.data import Dataset as TorchDataset
        
        if not isinstance(dataset, (TorchDataset, H5pyImageDataset)):
            raise ValueError("dataset must be a PyTorch Dataset object")
        
        # Set default batch size if not provided
        if batch_size is None:
            batch_size = 32
        
        num_samples = len(dataset)
        batch_size_norm = min(1024, num_samples)  # Use larger batches for normalization
        
        # Create DataLoader for normalization
        norm_dataloader = DataLoader(
            dataset,
            batch_size=batch_size_norm,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        # Initialize normalization - compute stats incrementally in batches to avoid loading everything
        print("Computing normalization statistics in batches...")
        
        # Get action dimension from first batch
        first_batch = next(iter(norm_dataloader))
        act_dim = first_batch['actions'].shape[1]
        
        
        # Compute normalization stats incrementally (online algorithm)
        # Initialize accumulators
        s_sum = torch.zeros(self.latent_dim).to(self.device)
        sp_sum = torch.zeros(self.latent_dim).to(self.device)
        a_sum = torch.zeros(act_dim).to(self.device)
        s_abs_sum = torch.zeros(self.latent_dim).to(self.device)
        a_abs_sum = torch.zeros(act_dim).to(self.device)
        out_abs_sum = torch.zeros(self.latent_dim).to(self.device)
        
        batch_data_list = []
        
        with torch.no_grad():
            for batch in tqdm(norm_dataloader, desc="Computing normalization statistics"):
                batch_images = batch['images'].float().to(self.device)
                batch_next_images = batch['next_images'].float().to(self.device)
                batch_actions = batch['actions'].float().to(self.device)
                
                s_latent_batch = self.encode_images(batch_images)
                sp_latent_batch = self.encode_images(batch_next_images)
                
                # Accumulate sums for mean computation
                s_sum += s_latent_batch.sum(dim=0)
                sp_sum += sp_latent_batch.sum(dim=0)
                a_sum += batch_actions.sum(dim=0)
                
                # Store batch data for MAD computation
                batch_data_list.append({
                    's_latent': s_latent_batch.cpu(),
                    'sp_latent': sp_latent_batch.cpu(),
                    'actions': batch_actions.cpu()
                })
        
        # Compute means
        s_shift = s_sum / num_samples
        sp_shift = sp_sum / num_samples
        a_shift = a_sum / num_samples
        out_shift = sp_shift - s_shift
        print('first batch done')
        with torch.no_grad():
            for batch_data in batch_data_list:
                s_latent_batch = batch_data['s_latent'].to(self.device)
                sp_latent_batch = batch_data['sp_latent'].to(self.device)
                batch_actions = batch_data['actions'].to(self.device)
                
                # Compute absolute deviations and accumulate
                s_abs_sum += torch.abs(s_latent_batch - s_shift).sum(dim=0)
                a_abs_sum += torch.abs(batch_actions - a_shift).sum(dim=0)
                out_abs_sum += torch.abs(sp_latent_batch - s_latent_batch - out_shift).sum(dim=0)
        
        # Compute scales
        s_scale = s_abs_sum / num_samples
        a_scale = a_abs_sum / num_samples
        out_scale = out_abs_sum / num_samples
        print('second batch done')
        # Set transformations
        if set_transformations:
            self.dynamics_net.set_transformations(s_shift, s_scale, a_shift, a_scale, 
                                                  out_shift, out_scale)
        else:
            s_shift = torch.zeros(self.latent_dim).to(self.device)
            s_scale = torch.ones(self.latent_dim).to(self.device)
            a_shift = torch.zeros(act_dim).to(self.device)
            a_scale = torch.ones(act_dim).to(self.device)
            out_shift = torch.zeros(self.latent_dim).to(self.device)
            out_scale = torch.ones(self.latent_dim).to(self.device)
        
        # Third pass: compute Y using the computed normalization stats
        # Reuse encoded latents from first pass to avoid re-encoding
        Y_list = []
        with torch.no_grad():
            for batch_data in batch_data_list:
                s_latent_batch = batch_data['s_latent'].to(self.device)
                sp_latent_batch = batch_data['sp_latent'].to(self.device)
                
                # Compute Y for this batch and store on CPU
                Y_batch = (sp_latent_batch - s_latent_batch - out_shift) / (out_scale + 1e-8)
                Y_list.append(Y_batch.cpu())
        
        # Concatenate Y from all batches (on CPU)
        Y = torch.cat(Y_list, dim=0)
        print('third batch done, training....')
        
        # Clear GPU memory
        #if self.device == 'cuda':
        #    torch.cuda.empty_cache()
        
        # Convert reconstruction_weights to tensor if provided
        if reconstruction_weights is not None:
            if isinstance(reconstruction_weights, np.ndarray):
                reconstruction_weights = torch.from_numpy(reconstruction_weights).float()
            elif not isinstance(reconstruction_weights, torch.Tensor):
                reconstruction_weights = torch.tensor(reconstruction_weights).float()
        
        # Check if perceptual loss should be used
        use_perceptual_loss = getattr(self.d_config, 'use_perceptual_loss', False)
        mse_weight = getattr(self.d_config, 'mse_weight', 0.85)
        perceptual_weight = getattr(self.d_config, 'perceptual_weight', 0.15)
        
        # Initialize perceptual loss function if enabled
        perceptual_loss_fn = None
        if use_perceptual_loss:
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision is required for perceptual loss. Install with: pip install torchvision")
            perceptual_loss_fn = PerceptualLoss(device=self.device).to(self.device)
            perceptual_loss_fn.eval()  # Set to eval mode
            for param in perceptual_loss_fn.parameters():
                param.requires_grad = False
        
        # Create Dreamer-style loss function
        # It will receive s_latent_batch (already encoded in forward pass)
        def dreamer_loss_fn(batch_X, Y_pred, Y_true, batch_weights=None):
            s_latent_batch, a_batch, next_images_batch = batch_X[0], batch_X[1], batch_X[2]
            
            # Check if Lipschitz constraints should be applied
            lipschitz_constraint = None
            soft_lipschitz_penalty_weight = None
            soft_lipschitz_sampling_eps = None
            soft_lipschitz_n_samples = None
            
            if hasattr(self.d_config, 'lipschitz_type') and self.d_config.lipschitz_type == "soft_sampling":
                lipschitz_constraint = getattr(self.d_config, 'lipschitz_constraint', None)
                soft_lipschitz_penalty_weight = getattr(self.d_config, 'soft_lipschitz_penalty_weight', None)
                soft_lipschitz_sampling_eps = getattr(self.d_config, 'soft_lipschitz_sampling_eps', None)
                soft_lipschitz_n_samples = getattr(self.d_config, 'soft_lipschitz_n_samples', None)
            
            # s_latent_batch is already encoded from forward pass, use it directly
            return dreamer_image_loss(
                (s_latent_batch, a_batch), Y_pred, Y_true,
                self.image_encoder, self.image_decoder,
                next_images_batch, out_shift, out_scale,
                reconstruction_weight=self.reconstruction_weight,
                dynamics_weight=self.dynamics_weight,
                dynamics_net=self.dynamics_net,
                lipschitz_constraint=lipschitz_constraint,
                soft_lipschitz_penalty_weight=soft_lipschitz_penalty_weight,
                soft_lipschitz_sampling_eps=soft_lipschitz_sampling_eps,
                soft_lipschitz_n_samples=soft_lipschitz_n_samples,
                reconstruction_weights=batch_weights,
                use_perceptual_loss=use_perceptual_loss,
                mse_weight=mse_weight,
                perceptual_weight=perceptual_weight,
                perceptual_loss_fn=perceptual_loss_fn
            )
        
        # Create a wrapper model that encodes images per-batch and uses (s, a) for forward pass
        # It stores s_latent so the loss function can access it
        class DynamicsNetWrapper(nn.Module):
            def __init__(self, dynamics_net, image_encoder):
                super().__init__()
                self.dynamics_net = dynamics_net
                self.image_encoder = image_encoder
                self.last_s_latent = None  # Store encoded latent for loss function
            
            def forward(self, images, a, *args):
                # Encode images to latent states (creates computation graph per-batch)
                s_latent = self.image_encoder(images)
                self.last_s_latent = s_latent  # Store for loss function
                # Use s and a for dynamics forward pass
                return self.dynamics_net.forward(s_latent, a)
        
        wrapped_model = DynamicsNetWrapper(self.dynamics_net, self.image_encoder)
        
        # Extract numpy arrays from dataset for faster direct indexing (avoids DataLoader overhead)
        from correct_il.datasets import ImageDynamicsDataset
        if isinstance(dataset, ImageDynamicsDataset):
            images_np = dataset.images
            next_images_np = dataset.next_images
            actions_np = dataset.actions
            # Extract weights if available
            if reconstruction_weights is None and hasattr(dataset, 'reconstruction_weights') and dataset.reconstruction_weights is not None:
                reconstruction_weights = dataset.reconstruction_weights
        elif isinstance(dataset, H5pyImageDataset) and dataset.load_into_memory:
            images_np = dataset.images
            next_images_np = dataset.next_images
            actions_np = dataset.actions
            # Extract weights if available
            if reconstruction_weights is None and hasattr(dataset, 'reconstruction_weights') and dataset.reconstruction_weights is not None:
                reconstruction_weights = dataset.reconstruction_weights
        else:
            # Fallback: convert dataset to numpy arrays
            print("Converting dataset to numpy arrays for faster training...")
            images_list = []
            next_images_list = []
            actions_list = []
            weights_list = []
            for i in range(len(dataset)):
                item = dataset[i]
                images_list.append(item['images'])
                next_images_list.append(item['next_images'])
                actions_list.append(item['actions'])
                if 'reconstruction_weights' in item:
                    weights_list.append(item['reconstruction_weights'])
            print('converting imgs')
            images_np = np.array(images_list, dtype=np.float32)
            print('converting next imgs')
            next_images_np = np.array(next_images_list, dtype=np.float32)
            print('converting actions')
            actions_np = np.array(actions_list, dtype=np.float32)
            # Extract weights if available
            if reconstruction_weights is None and len(weights_list) > 0:
                print('converting mask weights')
                reconstruction_weights = np.array(weights_list, dtype=np.float32)
        
        # Convert Y to numpy if needed
        Y_np = Y.numpy() if isinstance(Y, torch.Tensor) else Y
        
        # Create indices for shuffling
        indices = np.arange(num_samples)
        
        # Initialize wandb if requested
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=wandb_project or "ccil-dynamics",
                name=wandb_name,
                config={
                    "batch_size": batch_size,
                    "train_epochs": train_epochs,
                    "latent_dim": self.latent_dim,
                    "reconstruction_weight": self.reconstruction_weight,
                    "dynamics_weight": self.dynamics_weight,
                    "device": str(self.device),
                }
            )
        
        # Training loop using direct numpy array indexing (faster than DataLoader)
        epoch_losses = []
        steps_so_far = 0
        pbar = tqdm(range(train_epochs), desc="Training")
        
        for ep in pbar:
            ep_loss = defaultdict(int)
            num_batches = 0
            
            # Shuffle indices each epoch
            np.random.shuffle(indices)
            
            # Process in batches using direct numpy indexing
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_indices = indices[i:end_idx]
                
                # Direct numpy array indexing (much faster than Dataset/DataLoader)
                batch_images = torch.from_numpy(images_np[batch_indices]).float().to(self.device)
                batch_actions = torch.from_numpy(actions_np[batch_indices]).float().to(self.device)
                batch_next_images = torch.from_numpy(next_images_np[batch_indices]).float().to(self.device)
                batch_Y = torch.from_numpy(Y_np[batch_indices]).float().to(self.device)
                
                # Extract batch weights if reconstruction_weights is provided
                batch_weights = None
                if reconstruction_weights is not None:
                    batch_weights = torch.from_numpy(reconstruction_weights[batch_indices]).float().to(self.device)
               
                self.dynamics_opt.zero_grad()
                # Forward pass encodes images per-batch and stores s_latent
                Y_hat = wrapped_model.forward(batch_images, batch_actions, batch_next_images)
                # Get encoded latent from wrapper
                s_latent_batch = wrapped_model.last_s_latent
                # Pass s_latent to loss function instead of images
                batch_X_for_loss = (s_latent_batch, batch_actions, batch_next_images)
                
                loss_dict = dreamer_loss_fn(batch_X_for_loss, Y_hat, batch_Y, batch_weights=batch_weights)
                loss_dict['loss'].backward()
                self.dynamics_opt.step()
                loss_dict['loss'] = loss_dict['loss'].detach().to('cpu').numpy()
                for key, value in loss_dict.items():
                    if key == "mse_loss_tensor":
                        continue
                    ep_loss[key] += value
                num_batches += 1
                steps_so_far += 1
                
                # Log to wandb (per batch)
                if use_wandb and WANDB_AVAILABLE:
                    log_dict = {f"batch_{k}": v for k, v in loss_dict.items() if k != "mse_loss_tensor"}
                    # Ensure step is monotonically increasing
                    current_wandb_step = wandb.run.step if wandb.run else 0
                    log_step = max(current_wandb_step + 1, steps_so_far)
                    wandb.log(log_dict, step=log_step)
                
                if steps_so_far >= max_steps:
                    print("Number of grad steps exceeded threshold. Terminating early.")
                    break
            
            # Update tqdm description with loss values
            if len(ep_loss) > 0 and num_batches > 0:
                loss_str = ", ".join([f"{k}: {v/num_batches:.4f}" for k, v in ep_loss.items()])
                pbar.set_description(f"Epoch {ep}: {loss_str}")
                
                # Visualize reconstructions if requested
                if visualize_reconstructions and num_batches > 0 and ((ep+1) % visualize_interval == 0 or ep+1 == train_epochs):
                    # Use the first batch for visualization
                    with torch.no_grad():
                        # Get normalization parameters from dynamics_net
                        out_shift_vis = self.dynamics_net.out_shift
                        out_scale_vis = self.dynamics_net.out_scale
                        
                        # Get a sample batch using direct indexing
                        vis_indices = indices[:num_vis_samples]
                        vis_images = torch.from_numpy(images_np[vis_indices]).float().to(self.device)
                        vis_actions = torch.from_numpy(actions_np[vis_indices]).float().to(self.device)
                        vis_next_images = torch.from_numpy(next_images_np[vis_indices]).float().to(self.device)
                        
                        # Encode and predict
                        vis_s_latent = self.image_encoder(vis_images)
                        vis_Y_pred = self.dynamics_net.forward(vis_s_latent, vis_actions)
                        
                        # Denormalize predicted residual
                        vis_residual_pred = vis_Y_pred * (out_scale_vis + 1e-8) + out_shift_vis
                        vis_sp_latent_pred = vis_s_latent + vis_residual_pred
                        
                        # Decode predicted next images
                        vis_next_images_pred = self.image_decoder(vis_sp_latent_pred)
                        
                        # Convert to numpy for visualization
                        vis_images_np = vis_images.cpu().numpy()
                        vis_next_images_np = vis_next_images.cpu().numpy()
                        vis_next_images_pred_np = vis_next_images_pred.cpu().numpy()
                        
                        # Handle image format conversion for visualization
                        if vis_next_images_pred_np.shape[1] in [1, 3, 6] and len(vis_next_images_pred_np.shape) == 4:
                            # (B, C, H, W) -> (B, H, W, C)
                            vis_next_images_pred_np = vis_next_images_pred_np.transpose(0, 2, 3, 1)
                        
                        # Create visualization grid
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(num_vis_samples, 3, figsize=(12, 4*num_vis_samples))
                        if num_vis_samples == 1:
                            axes = axes.reshape(1, -1)
                        
                        for i in range(num_vis_samples):
                            # Current image
                            if vis_images_np.shape[-1] == 6:
                                # Split 6-channel into two 3-channel images for display
                                axes[i, 0].imshow(vis_images_np[i, :, :, :3])
                                axes[i, 0].set_title(f'Sample {i+1}: Current (agentview)')
                            else:
                                axes[i, 0].imshow(vis_images_np[i])
                                axes[i, 0].set_title(f'Sample {i+1}: Current')
                            axes[i, 0].axis('off')
                            
                            # Ground truth next image
                            if vis_next_images_np.shape[-1] == 6:
                                axes[i, 1].imshow(vis_next_images_np[i, :, :, :3])
                                axes[i, 1].set_title(f'Ground Truth Next (agentview)')
                            else:
                                axes[i, 1].imshow(vis_next_images_np[i])
                                axes[i, 1].set_title('Ground Truth Next')
                            axes[i, 1].axis('off')
                            
                            # Predicted next image
                            if vis_next_images_pred_np.shape[-1] == 6:
                                axes[i, 2].imshow(vis_next_images_pred_np[i, :, :, :3])
                                axes[i, 2].set_title(f'Predicted Next (agentview)')
                            else:
                                axes[i, 2].imshow(vis_next_images_pred_np[i])
                                axes[i, 2].set_title('Predicted Next')
                            axes[i, 2].axis('off')
                        
                        plt.tight_layout()
                        
                        # Log to wandb if available
                        if use_wandb and WANDB_AVAILABLE:
                            # Ensure step is monotonically increasing by using max of wandb's current step and our step
                            current_wandb_step = wandb.run.step if wandb.run else 0
                            log_step = max(current_wandb_step + 1, steps_so_far)
                            wandb.log({f"reconstructions epoch {ep+1}": wandb.Image(fig)}, step=log_step)
                        
                        # Save to file
                        import os
                        if vis_output_dir is None:
                            vis_dir = os.path.join(os.getcwd(), "reconstruction_vis")
                        else:
                            vis_dir = vis_output_dir
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_path = os.path.join(vis_dir, f"epoch_{ep:04d}.png")
                        plt.savefig(vis_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                       # print(f"Saved reconstruction visualization to {vis_path}")
                
                # Log to wandb (per epoch)
                if use_wandb and WANDB_AVAILABLE:
                    log_dict = {f"epoch_{k}": v/num_batches for k, v in ep_loss.items()}
                    log_dict["epoch"] = ep
                    # Ensure step is monotonically increasing by using max of wandb's current step and our step
                    current_wandb_step = wandb.run.step if wandb.run else 0
                    log_step = max(current_wandb_step + 1, steps_so_far)
                    wandb.log(log_dict, step=log_step)
                
                # Save checkpoint every checkpoint_interval epochs
                if checkpoint_dir is not None and (ep + 1) % checkpoint_interval == 0:
                    import os
                    import pickle
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{ep+1:04d}.pkl")
                    checkpoint_data = {
                        'step': steps_so_far,
                        'epoch': ep + 1,
                        'model_state_dict': {
                            'image_encoder': self.image_encoder.state_dict(),
                            'image_decoder': self.image_decoder.state_dict(),
                            'dynamics_net': self.dynamics_net.state_dict(),
                        },
                        'optimizer_state_dict': self.dynamics_opt.state_dict(),
                        'epoch_losses': epoch_losses,
                    }
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    print(f"Saved checkpoint to {checkpoint_path}")
            
            epoch_losses.append(ep_loss)
            
            if steps_so_far >= max_steps:
                break
        
        # Save final checkpoint (if not already saved at the last checkpoint interval)
        if checkpoint_dir is not None:
            import os
            import pickle
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Only save final checkpoint if it wasn't already saved at the checkpoint interval
            if train_epochs % checkpoint_interval != 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final_epoch_{train_epochs:04d}.pkl")
                checkpoint_data = {
                    'step': steps_so_far,
                    'epoch': train_epochs,
                    'model_state_dict': {
                        'image_encoder': self.image_encoder.state_dict(),
                        'image_decoder': self.image_decoder.state_dict(),
                        'dynamics_net': self.dynamics_net.state_dict(),
                    },
                    'optimizer_state_dict': self.dynamics_opt.state_dict(),
                    'epoch_losses': epoch_losses,
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"Saved final checkpoint to {checkpoint_path}")
        
        # Finish wandb run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        num_batches_per_epoch = (num_samples + batch_size - 1) // batch_size
        epoch_losses = list_of_dict_to_dict_of_list(epoch_losses, float(num_batches_per_epoch))
        return epoch_losses
    
    @torch.no_grad()
    def eval_prediction_error(self, dataset, batch_size):
        """
        Evaluate prediction error using Dreamer-style loss (both components).
        
        Args:
            dataset: Dataset object containing images, actions, next_images
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with 'total_error', 'dynamics_error', 'reconstruction_error'
        """
        from torch.utils.data import DataLoader
        
        eval_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        dynamics_errors = []
        reconstruction_errors = []
        
        dynamics_weight = self.dynamics_weight
        reconstruction_weight = self.reconstruction_weight
        
        # Get normalization stats
        s_shift = self.dynamics_net.s_shift
        s_scale = self.dynamics_net.s_scale
        a_shift = self.dynamics_net.a_shift
        a_scale = self.dynamics_net.a_scale
        out_shift = self.dynamics_net.out_shift
        out_scale = self.dynamics_net.out_scale
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch_images = batch['images'].float().to(self.device)
                batch_actions = batch['actions'].float().to(self.device)
                batch_next_images = batch['next_images'].float().to(self.device)
                
                # Encode images to latent states
                s_latent = self.image_encoder(batch_images)
                sp_latent_true = self.image_encoder(batch_next_images)
                
                # Normalize inputs for dynamics net
                s_latent_norm = (s_latent - s_shift) / (s_scale + 1e-8)
                a_norm = (batch_actions - a_shift) / (a_scale + 1e-8)
                
                # Compute true normalized residual
                residual_true = sp_latent_true - s_latent
                residual_true_norm = (residual_true - out_shift) / (out_scale + 1e-8)
                
                # Predict normalized residual using dynamics net
                residual_pred_norm = self.dynamics_net.forward(s_latent_norm, a_norm)
                
                # 1. Dynamics loss: MSE in normalized latent space
                dynamics_loss = F.mse_loss(residual_pred_norm, residual_true_norm, reduction='none')
                dynamics_loss = dynamics_loss.mean(dim=1)  # Average over latent dimensions
                dynamics_errors += dynamics_loss.cpu().tolist()
                
                # 2. Reconstruction loss: decode and compare images
                # Denormalize predicted residual
                residual_pred = residual_pred_norm * (out_scale + 1e-8) + out_shift
                sp_latent_pred = s_latent + residual_pred
                
                # Decode predicted next images
                next_images_pred = self.image_decoder(sp_latent_pred)
                
                # Handle image format conversion
                if batch_next_images.dim() == 4:
                    if batch_next_images.shape[-1] in [1, 3, 6]:  # (B, H, W, C) format
                        if next_images_pred.shape[1] in [1, 3, 6]:
                            next_images_pred = next_images_pred.permute(0, 2, 3, 1)
                    else:  # (B, C, H, W) format
                        if next_images_pred.shape[-1] in [1, 3, 6]:
                            next_images_pred = next_images_pred.permute(0, 3, 1, 2)
                
                # Reconstruction loss: pixel-wise MSE
                reconstruction_loss = F.mse_loss(next_images_pred, batch_next_images, reduction='none')
                reconstruction_loss = reconstruction_loss.flatten(start_dim=1).mean(dim=1)  # Average over pixels
                reconstruction_errors += reconstruction_loss.cpu().tolist()
        
        # Compute mean errors
        mean_dynamics_error = np.mean(dynamics_errors)
        mean_reconstruction_error = np.mean(reconstruction_errors)
        total_error = dynamics_weight * mean_dynamics_error + reconstruction_weight * mean_reconstruction_error
        
        return {
            'total_error': total_error,
            'dynamics_error': mean_dynamics_error,
            'reconstruction_error': mean_reconstruction_error
        }
    
    @torch.no_grad()
    def eval_lipschitz_coeff(self, images, actions, batch_size=None):
        """
        Evaluate Lipschitz coefficients for image-based model.
        Encodes images to latent states first, then computes Lipschitz coefficients.
        """
        batch_size = batch_size if batch_size else len(images)
        
        # Encode images to latent states
        s_latent = self.encode_images(images)
        a = torch.as_tensor(actions).float().to(self.device)

        def predict_from_latent(s_lat, a_lat):
            """Predict next latent state from latent state and action."""
            s_lat = torch.as_tensor(s_lat).float().to(self.device)
            a_lat = torch.as_tensor(a_lat).float().to(self.device)
            return s_lat + self.dynamics_net.predict(s_lat, a_lat)
        
        def predict_concat(state_action):
            obs = state_action[...,:s_latent.shape[-1]]
            act = state_action[...,s_latent.shape[-1]:]
           
            return predict_from_latent(obs, act)
        
        
        # Compute batched jacobian with vectorization
        import functorch
        jac_fn = functorch.vmap(functorch.jacrev(predict_concat))
        
        s_a = np.concatenate([s_latent.cpu().numpy(), a.cpu().numpy()], axis=-1)
        lipschits_coeffs = []
        for i in tqdm(range(0, len(s_latent), batch_size)):
            batch = torch.as_tensor(s_a[i:i+batch_size], dtype=torch.float32, device=self.device)
            jacs = jac_fn(batch)
            assert jacs.shape == (batch.shape[0], s_latent.shape[-1], batch.shape[-1])
            local_L = torch.linalg.norm(jacs, ord=2, dim=(-2,-1)).cpu().numpy()
            lipschits_coeffs.append(local_L)
        return np.concatenate(lipschits_coeffs, axis=0)
    
    def to(self, device):
        self.image_encoder.to(device)
        self.image_decoder.to(device)
        self.dynamics_net.to(device)
        self.device = device
    
    def is_cuda(self):
        return next(self.dynamics_net.parameters()).is_cuda

class DynamicsNet(nn.Module):
    def __init__(self, state_dim, act_dim,
                 hidden_size=(64,64),
                 s_shift = None,
                 s_scale = None,
                 a_shift = None,
                 a_scale = None,
                 out_shift = None,
                 out_scale = None,
                 out_dim = None,
                 use_mask = True,
                 ):
        super(DynamicsNet, self).__init__()

        self.state_dim, self.act_dim, self.hidden_size = state_dim, act_dim, hidden_size
        self.out_dim = state_dim if out_dim is None else out_dim
        self.layer_sizes = (state_dim + act_dim, ) + tuple(hidden_size) + (self.out_dim, )
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                    for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu
        self.use_mask = use_mask
        self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)

    def enable_spectral_normalization(self, lipschitz_constraint):
        _apply_spectral_normalization_recursively(self.fc_layers, lipschitz_constraint)

    def set_transformations(self, s_shift=None, s_scale=None,
                            a_shift=None, a_scale=None,
                            out_shift=None, out_scale=None):
        if s_shift is None:
            self.s_shift     = torch.zeros(self.state_dim)
            self.s_scale    = torch.ones(self.state_dim)
            self.a_shift     = torch.zeros(self.act_dim)
            self.a_scale    = torch.ones(self.act_dim)
            self.out_shift   = torch.zeros(self.out_dim)
            self.out_scale  = torch.ones(self.out_dim)
        elif type(s_shift) == torch.Tensor:
            self.s_shift, self.s_scale = s_shift, s_scale
            self.a_shift, self.a_scale = a_shift, a_scale
            self.out_shift, self.out_scale = out_shift, out_scale
        elif type(s_shift) == np.ndarray:
            self.s_shift     = torch.from_numpy(np.float32(s_shift))
            self.s_scale    = torch.from_numpy(np.float32(s_scale))
            self.a_shift     = torch.from_numpy(np.float32(a_shift))
            self.a_scale    = torch.from_numpy(np.float32(a_scale))
            self.out_shift   = torch.from_numpy(np.float32(out_shift))
            self.out_scale  = torch.from_numpy(np.float32(out_scale))
        else:
            print("Unknown type for transformations")
            quit()

        device = next(self.parameters()).data.device
        self.s_shift, self.s_scale = self.s_shift.to(device), self.s_scale.to(device)
        self.a_shift, self.a_scale = self.a_shift.to(device), self.a_scale.to(device)
        self.out_shift, self.out_scale = self.out_shift.to(device), self.out_scale.to(device)
        # if some state dimensions have very small variations, we will force it to zero
        self.mask = self.out_scale >= 1e-8

        self.transformations = dict(s_shift=self.s_shift, s_scale=self.s_scale,
                                    a_shift=self.a_shift, a_scale=self.a_scale,
                                    out_shift=self.out_shift, out_scale=self.out_scale)

    def forward(self, s, a):
        # Given raw input return the normalized residual
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        # normalize inputs
        s_in = (s - self.s_shift)/(self.s_scale + 1e-8)
        a_in = (a - self.a_shift)/(self.a_scale + 1e-8)
        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out

    def predict(self, s, a):
        # Given raw input return the (unnormalized) residual
        out = self.forward(s, a)
        out = out * (self.out_scale + 1e-8) + self.out_shift
        out = out * self.mask if self.use_mask else out
        return out

    def get_params(self):
        network_weights = [p.data for p in self.parameters()]
        transforms = (self.s_shift, self.s_scale,
                      self.a_shift, self.a_scale,
                      self.out_shift, self.out_scale)
        return dict(weights=network_weights, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = new_params['transforms']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)


def fit_model(nn_model, X, Y, optimizer, loss_fn,
              batch_size, epochs, max_steps=1e10):
    """
    :param nn_model:        pytorch model of form Y = f(*X) (class)
    :param X:               tuple of necessary inputs to the function
    :param Y:               desired output from the function (tensor)
    :param optimizer:       optimizer to use
    :param loss_func:       loss criterion
    :param batch_size:      mini-batch size
    :param epochs:          number of epochs
    :return:
    """

    assert type(X) == tuple
    for d in X: assert type(d) == torch.Tensor
    assert type(Y) == torch.Tensor
    device = Y.device
    for d in X: assert d.device == device

    num_samples = Y.shape[0]
    num_steps = int(num_samples // batch_size)
    epoch_losses = []
    steps_so_far = 0
    # set description to loss value for each key in ep_loss
    pbar = tqdm(range(epochs), desc="Training")
    for ep in pbar:
        rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
        ep_loss = defaultdict(int)

        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_X  = [d[data_idx] for d in X]
            batch_Y  = Y[data_idx]
            optimizer.zero_grad()
            Y_hat    = nn_model.forward(*batch_X)
           
            loss_dict = loss_fn(batch_X, Y_hat, batch_Y)
            loss_dict['loss'].backward()
            optimizer.step()
            loss_dict['loss'] = loss_dict['loss'].detach().to('cpu').numpy()
            for key, value in loss_dict.items():
                if key == "mse_loss_tensor":
                    continue
                ep_loss[key] += value
        
        # Update tqdm description with loss values
        if len(ep_loss) > 0:
            loss_str = ", ".join([f"{k}: {v/num_steps:.4f}" for k, v in ep_loss.items()])
            pbar.set_description(f"Epoch {ep}: {loss_str}")
        
        epoch_losses.append(ep_loss)
        steps_so_far += num_steps
        if steps_so_far >= max_steps:
            print("Number of grad steps exceeded threshold. Terminating early.")
            break
    epoch_losses = list_of_dict_to_dict_of_list(epoch_losses, float(num_steps))
    return epoch_losses

def list_of_dict_to_dict_of_list(a_list, divisor=1.0):
    if len(a_list) == 0:
        return []
    keys = a_list[0].keys()
    new_list = {k:[] for k in keys}
    for a_dict in a_list:
        for k in keys:
            new_list[k].append(a_dict[k] / divisor)
    return new_list
