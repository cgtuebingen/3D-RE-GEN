import torch
import torch.nn as nn

# datastructures
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance

# 3D transformations functions
from pytorch3d.transforms import  so3_exponential_map, Transform3d
import torch.nn.functional as F


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


def focal_loss(inputs, targets, alpha=0.5, gamma=2.0):
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


# setup differential model ####################################################
# Updated Model Class
class Model(nn.Module):
    def __init__(self,
                 meshes,
                 renderer,

                 plane_to_world_transform,  # <-- NEW: Pass in the transform
                 target_pointcloud=None,
                 cameras=None,
                 config: dict = None,
                 **kwargs):
        super().__init__()
        
        # ... (all your existing initializations like self.meshes, self.renderer, etc.)
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
        
        # ✅ CRITICAL: We need to work in PLANE SPACE, not world space!
        # First, transform the initial mesh vertices to plane space
        world_to_plane = plane_to_world_transform.inverse()
        verts_in_plane = world_to_plane.transform_points(verts)
        
        # ✅ FIX: Calculate pivot in PLANE space
        # Plane coords: X and Z are tangent to floor, Y is perpendicular (up/down)
        # Pivot should be at bottom-center: center in X-Z, minimum in Y
        verts_min = verts_in_plane.min(dim=0).values
        verts_max = verts_in_plane.max(dim=0).values
        
        self.pivot_plane = torch.tensor([
            (verts_min[0] + verts_max[0]) / 2.0,  # X center (tangent to plane)
            verts_min[1],                          # Y minimum (should be 0 to sit on floor!)
            (verts_min[2] + verts_max[2]) / 2.0   # Z center (tangent to plane)
        ], device=self.device)
        
        print(f"[DEBUG] Object pivot in plane space: {self.pivot_plane}")
        print(f"[DEBUG] Object Y-range in plane space: [{verts_min[1]:.4f}, {verts_max[1]:.4f}]")
        print(f"[DEBUG] Expected: Y_min should be ~0 to sit on floor")
        
        # ✅ Store vertices centered at pivot in plane space
        self.register_buffer("base_verts_plane", verts_in_plane - self.pivot_plane)
        
        # ✅ IMPORTANT: The pivot Y should be 0 (on the floor surface)
        # We'll enforce this by always placing pivot at Y=0 in plane space
        self.register_buffer("pivot_plane_centered", torch.tensor([
            self.pivot_plane[0],  # Keep X position
            0.0,                   # Force Y=0 (on floor)
            self.pivot_plane[2]   # Keep Z position
        ], device=self.device))
        
        self.mask = kwargs.get('mask', None)
        self.register_buffer("faces", meshes.faces_list()[0])
        self.textures = meshes.textures
        self.renderer = renderer

        self.cameras = cameras
        if self.cameras is None:
            raise ValueError("Cameras not provided")
        self.register_buffer('image_ref', torch.from_numpy(self.mask))
        #self.register_buffer('target_pointcloud', target_pointcloud)
        
        self.target_pcl = target_pointcloud #Pointclouds(points=[target_pointcloud])
        self.max_it = config.get("max_iterations", 100)

        # add bbox if provided
        if 'background_bbox' in kwargs and kwargs['background_bbox'] is not None:
            self.register_buffer('background_bbox', torch.tensor(kwargs['background_bbox'], device=self.device))
  
        # ...

        # --- MODIFICATION 1: Store the plane transformation ---
        self.register_buffer('plane_to_world_matrix', plane_to_world_transform.get_matrix())

        # --- MODIFICATION 2: Change the learnable parameters ---
        # The object now only moves in 2D (u,v) on the plane. Z is fixed at 0.
        self.translation_uv = nn.Parameter(
            torch.zeros(1, 2, device=self.device)
        )
        
        # The object only rotates around the plane's normal (the new Z-axis).
        self.rotation_yaw = nn.Parameter(
            torch.zeros(1, device=self.device)
        )

        # Uniform scale remains the same.
        self.scale = nn.Parameter(torch.ones(1, 1, device=self.device))

        # We no longer need the old 3D translation and rotation
        # self.translation = ... (REMOVED)
        # self.rotation = ... (REMOVED)

    def forward(self):
        self.iteration += 1

        # --- CORRECT PLANE-BASED TRANSFORMATION ---
        
        # ✅ STEP 1: Scale uniformly from bottom-up (pivot at base)
        # base_verts_plane are already centered at pivot, so scaling happens naturally
        scaled_verts = self.base_verts_plane * self.scale
        
        annealing_duration = self.max_it * 3/4
        # The noise_factor will go from 1.0 to 0.0 linearly.

        rsm = self.config.get("rotation_speed_mult", 20.0)
        #anneal_rot_speed = max(1.0, rsm - (self.iteration / annealing_duration) * rsm)
        # This makes the object spin on the floor
        axis_angle = torch.zeros(1, 3, device=self.device)
        axis_angle[:, 1] = self.rotation_yaw  # Y-axis in plane space
        R_plane = so3_exponential_map(axis_angle * rsm )  # Shape: (1, 3, 3)
        # R_plane = R_plane @ R_exploratory
        # Rotation: need to handle batch dimension properly
        # scaled_verts is (N, 3), R_plane is (1, 3, 3)
        rotated_verts = torch.matmul(scaled_verts.unsqueeze(0), R_plane.transpose(1, 2)).squeeze(0)
        
        # ✅ STEP 3: Translate to position on floor
        # translation_uv = (X, Z) movement along the floor
        # We construct (X, 0, Z) to keep object ON the floor surface
        floor_position = torch.cat([
            self.translation_uv[:, 0:1],           # X position (along floor)
            torch.zeros(1, 1, device=self.device), # Y = 0 (ON the floor)
            self.translation_uv[:, 1:2]            # Z position (along floor)
        ], dim=1)  # Shape: (1, 3)
        
        # Add floor position to pivot position
        final_position_plane = self.pivot_plane_centered.unsqueeze(0) + floor_position  # (1, 3)
        translated_verts = rotated_verts + final_position_plane  # Broadcasting: (N, 3) + (1, 3) = (N, 3)
        
        # ✅ STEP 4: Transform from plane space back to world space
        plane_to_world = Transform3d(matrix=self.plane_to_world_matrix, device=self.device)
        # Add batch dimension for transform_points, then squeeze it back
        transformed_verts = plane_to_world.transform_points(translated_verts.unsqueeze(0)).squeeze(0)
        
        #################################################################################
        # THE REST OF YOUR FORWARD PASS REMAINS EXACTLY THE SAME
        #################################################################################

        current_mesh = Meshes(
            verts=[transformed_verts],
            faces=[self.faces],
            textures=self.textures,
        )

        image = self.renderer(meshes_world=current_mesh)


        P = image[..., 3].squeeze(0)  # Extract alpha channel as silhouette
        P = torch.sigmoid(P)

        P = P.squeeze(0)
        T = self.image_ref.to(self.device)
        T = (T > 0.5).float() if T.max() > 1 else T  # Ensure binary

        # Compute silhouette loss using focal loss
        focal_bce_loss = focal_loss(P, T)
        dice_loss = 1 - (2 * (P * T).sum() + 1e-6) / (P.sum() + T.sum() + 1e-6)
        loss_silhouette = 0.75 * dice_loss + 0.25 * focal_bce_loss
        
        #loss_silhouette = ((P - T)**2).mean() #+ 0.5 * ((P**2) * (1-T)).mean() # Focal loss component
        # print(f"[DEBUG] Iteration {self.iteration}: Silhouette Loss = {(loss_silhouette * self.silhoutte_loss).item():.6f}")

        loss_3d = point_mesh_face_distance(current_mesh, self.target_pcl, min_triangle_area=1e-7)

        # Unpack mesh samples and normals
        # mesh_samples, mesh_normals = sample_points_from_meshes(current_mesh, 5000, return_normals=True)

        # If your target point cloud is a Pointclouds object with normals:
        # target_points = self.target_pcl.points_padded()  # (batch, N, 3)
        # target_normals = self.target_pcl.normals_padded()  # (batch, N, 3)

        # Compute Chamfer loss with normals
        # chamfer_loss_p, chamfer_loss_q = chamfer_distance(
        #     x=mesh_samples, y=target_points,
        #     x_normals=mesh_normals, y_normals=target_normals,
        #     point_reduction="mean"
        # )
        #print(f"[DEBUG] Iteration {self.iteration}: Chamfer Loss P = {chamfer_loss_p.item():.6f}, Q = {chamfer_loss_q.item():.6f}")
        #chamfer_loss = chamfer_loss_p * 10 + chamfer_loss_q * 0.25

        #centroid_loss = centroid_loss(mesh_pc, self.target_)
        #loss_3d =  chamfer_loss

        # loss_3d over time
        # over_time_mult = max(1.0, self.iteration / self.max_it)
        # loss_3d *= over_time_mult
        # print(f"[DEBUG] Iteration {self.iteration}: 3D Loss = {(loss_3d * self.loss_3d_mult).item():.6f}")
        loss_bbox = 0.0
        # bbox loss if bbox provided
        if hasattr(self, 'background_bbox'):
            verts = current_mesh.verts_padded()[0]
            loss_bbox = bounding_box_loss(verts, self.background_bbox) # weight for bbox loss

        #print(f"[DEBUG] Iteration {self.iteration}: Silhouette Loss = {loss_silhouette.item():.6f}, 3D Loss = {loss_bbox_mult.item():.6f}" + (f", BBox Loss = {loss_bbox.item():.6f}" if hasattr(self, 'background_bbox') else ""))

        total_loss = (
            self.silhoutte_loss * loss_silhouette
            + self.loss_3d_mult * loss_3d
            + self.loss_bbox_mult * loss_bbox 
        )

        return total_loss, current_mesh, P
