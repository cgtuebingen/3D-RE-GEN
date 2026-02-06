import torch
import torch.nn as nn

# datastructures
from pytorch3d.structures import Meshes, Pointclouds

from pytorch3d.loss import chamfer_distance, point_mesh_face_distance

# 3D transformations functions
from pytorch3d.transforms import  so3_exponential_map
import logging

# from utils.metrics import (
#     compute_chamfer_distance,
#     compute_fscore,
#     compute_volume_iou,
#     icp,
# )
from scipy.stats import wasserstein_distance

def bounding_box_loss(verts, bbox, ignore_axis=1):
    min_xyz = torch.tensor(bbox[:3], device=verts.device)
    max_xyz = torch.tensor(bbox[3:], device=verts.device)
    below_min = torch.relu(min_xyz - verts)
    above_max = torch.relu(verts - max_xyz)
    # Create a mask to zero out the floor axis (Y=1)
    mask = torch.ones_like(below_min)
    mask[:, ignore_axis] = 0
    penalty = (below_min + above_max) * mask
    return penalty.sum() / verts.shape[0]



def symmetric_chamfer_loss(mesh_pts: torch.Tensor,
                           target_pts: torch.Tensor,
                           point_reduction: str = "mean") -> torch.Tensor:
    """
    mesh_pts   – (M, 3)   points sampled from the current mesh (world space)
    target_pts – (N, 3)   fixed point cloud obtained from the depth map
    Returns a **single scalar** Chamfer distance that already contains
    both src→tgt and tgt→src terms.
    """
    # Add the batch dimension required by the API
    src = mesh_pts.unsqueeze(0)   # (1, M, 3)
    tgt = target_pts.unsqueeze(0) # (1, N, 3)

    # `chamfer_distance` returns (loss, loss_normals).  We only need the loss.
    loss, _ = chamfer_distance(
        src,
        tgt,
        point_reduction=point_reduction,   # "mean" → a single scalar
        batch_reduction="mean",            # average over the (single) batch
        single_directional=False,          # default → symmetric loss
        norm=2,                            # L2 distance (you can change to 1)
        
    )
    return loss


# simple centroid loss
def centroid_loss(mesh_pts: torch.Tensor, target_pts: torch.Tensor) -> torch.Tensor:
    """
    Compute the centroid loss between the mesh points and the target points.
    """
    mesh_centroid = mesh_pts.mean(dim=0)
    target_centroid = target_pts.mean(dim=0)
    return nn.functional.mse_loss(mesh_centroid, target_centroid)


# setup differential model ####################################################
class Model(nn.Module):
    """
    A model that takes a mesh and a reference image, and optimizes the mesh to match the silhouette of the reference image.
    Also takes in depth information to constrain the optimization.
    """

    def __init__(self, 
                    meshes, 
                    renderer, 

                    target_pointcloud=None,
                    cameras=None,
                    config: dict = None, 
                    **kwargs):
        super().__init__()

        self.meshes = meshes
        self.device = meshes.device
        self.iteration = 0

        self.config = config
        if self.config is None:
            raise ValueError("Config must be provided in kwargs")

        self.silhoutte_loss = self.config.get("silhoutte_loss", 1.0)
        self.loss_3d_mult = self.config.get("loss_3d", 1.0)
        self.loss_bbox_mult = self.config.get("loss_bbox", 1.0)

        verts = meshes.verts_list()[0]
        self.center = verts.mean(dim=0)

        # Set mask from kwargs
        self.mask = kwargs.get('mask', None)

        # Store original mesh components
        self.register_buffer("base_verts", meshes.verts_list()[0])
        self.register_buffer("faces", meshes.faces_list()[0])
        self.textures = meshes.textures

        self.renderer = renderer # silhoutte renderer


        self.cameras = cameras
        # Check if cameras is None and raise error if so
        if self.cameras is None:
            raise ValueError("Cameras not provided")

        self.register_buffer('image_ref', torch.from_numpy(self.mask))
        #self.register_buffer('depth_refObj_masked', torch.from_numpy(depth_refObj_masked))
        self.register_buffer('target_pointcloud', target_pointcloud)

        # add pointcloud object
        self.target_pcl = Pointclouds(points=[target_pointcloud])

        self.model_screen_pos = kwargs.get('model_screen_pos', (0.5, 0.5)) # x,y in normalized screen space 0..1
        logging.info(f"Target 2D Centroid x, y: {self.model_screen_pos[0]:.4f}, {self.model_screen_pos[1]:.4f}")

        # for image centroid loss
        try:
            H, W = self.mask.shape
        except Exception:
            H, W = self.image_ref.shape[-2], self.image_ref.shape[-1]
        x_grid = torch.arange(W, dtype=torch.float32, device=self.device).unsqueeze(0).expand(H, W) / max(W - 1, 1)
        y_grid = torch.arange(H, dtype=torch.float32, device=self.device).unsqueeze(1).expand(H, W) / max(H - 1, 1)
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)

        if 'background_bbox' in kwargs and kwargs['background_bbox'] is not None:
            self.register_buffer('background_bbox', torch.tensor(kwargs['background_bbox'], device=self.device))
  


        # Initialize transformation parameters # Translation (learnable)
        self.translation = nn.Parameter(
            torch.rand(1, 3, device=self.device) * 0.01)

        # Axis-angle rotation (learnable)
        
        if self.config.get("use_5DOF", True):
            self.rotation = nn.Parameter(
                torch.zeros(1, device=self.device))
        else:
            self.rotation = nn.Parameter(
                torch.zeros(1, 3, device=self.device))
            # only allow rotation around Y
        

        # Uniform scale (learnable)
        self.scale = nn.Parameter(torch.ones(1, 1, device=self.device))

        # Freeze
        self.scale.requires_grad = True
        self.translation.requires_grad = True
        self.rotation.requires_grad = True



    def forward(self):
        self.iteration += 1

        #### 1. DEFINE ANNEALING SCHEDULE #######################################################################
        # Anneal the exploratory rotation over, e.g., the first 500 iterations
        annealing_duration = 200
        # The noise_factor will go from 1.0 to 0.0 linearly.
        noise_factor = max(0.0, 1.0 - self.iteration / annealing_duration) * 0.0

        #### 2. APPLY EXPLORATORY ROTATION #####################################################################
        # This rotation is NOT learned. It just perturbs the mesh.
        # We'll create a simple, fixed-axis rotation that swings the object around.
        # The angle of this swing decreases over time thanks to noise_factor.
        exploratory_angle = torch.tensor([3.14159], device=self.device) * noise_factor # Start with a 180-degree swing
        exploratory_axis_angle = torch.zeros(1, 3, device=self.device)
        exploratory_axis_angle[:, 1] = exploratory_angle # Rotate around Y-axis
        
        R_exploratory = so3_exponential_map(exploratory_axis_angle)[0]

        #### 3. APPLY LEARNABLE ROTATION (YOUR EXISTING CODE) ################################################
        if self.config.get("use_5DOF", True):
            theta = self.rotation.view(1)
            axis_angle = torch.zeros(1, 3, device=theta.device, dtype=theta.dtype)
            axis_angle[:, 1] = theta
            # I removed the '* 20' multiplier. A high LR is better handled by the optimizer.
            R_learnable = so3_exponential_map(axis_angle)[0] 
        else:
            R_learnable = so3_exponential_map(self.rotation)[0]

        # Combine the rotations: first the exploratory, then the learned one
        R_transform = R_learnable @ R_exploratory


        centered_verts = self.base_verts - self.center
        rotated_verts = (R_transform @ centered_verts.T).T # torch.matmul(centered_verts, R_transform.T) #(R_transform @ centered_verts.T).T
        scaled_rotated_verts = self.scale * rotated_verts
        transformed_verts = scaled_rotated_verts + self.center + self.translation
        #########################################################################################################

        # Uses Meshes() instead
        current_mesh = Meshes(
            verts=[transformed_verts],
            faces=[self.faces],
            textures=self.textures,  # Use preserved textures
            
        )

        image = self.renderer(
            meshes_world=current_mesh,  
        ) # silhouette of mesh


        
        P = image[..., 3].squeeze(0)  # Extract alpha channel as silhouette
        P = torch.sigmoid(P)

        P = P.squeeze(0)
        T = self.image_ref.to(self.device)
        T = (T > 0.5).float() if T.max() > 1 else T  # Ensure binary

        #loss_silhouette = ((P - T)**2).mean()
        dice_loss = 1 - (2 * (P * T).sum() + 1e-6) / (P.sum() + T.sum() + 1e-6)
        bce_loss = nn.functional.binary_cross_entropy(P, T)
        loss_silhouette = 0.75 * dice_loss + 0.25 * bce_loss
        

        loss_3d = point_mesh_face_distance(current_mesh, self.target_pcl)
        logging.debug(f"Chamfer Loss: {loss_3d.item()}")

        loss_bbox = 0.0
        if hasattr(self, 'background_bbox'):
            verts = current_mesh.verts_padded()[0]
            loss_bbox = bounding_box_loss(verts, self.background_bbox)


        # TODO fix silhouette loss
        # Combine losses (adjust weights as needed)
        total_loss = (
            self.silhoutte_loss * loss_silhouette
            # + self.loss3d_mult *loss_3d
            + loss_3d * self.loss_3d_mult
            + self.loss_bbox_mult * loss_bbox 

        )

        return total_loss, current_mesh, P
