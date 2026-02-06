import argparse
import math

import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import os,sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dust3r")))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl
pl.ion()

from global_utils import load_config

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)

    # Save camera parameters separately (only the first camera for simplicity)
    # Keeping the original comment formatting
    # new !!
    # add camera_angle_x to the saved parameters
    image_size = imgs[0].shape[1]
    print('image_size', image_size)
    camera_angle_x = float(2.0 * np.arctan(image_size / (2.0 * focals[0])))
    print('camera_angle_x in rad, deg: ', camera_angle_x, np.rad2deg(camera_angle_x))

    cam_outfile = os.path.join(outdir, 'camera.npz')
    if not silent:
        print('(exporting camera parameters to', cam_outfile, ')')
    cam_dict = {
        "extrinsic": cams2world[0],
        "focal": focals[0],
        "image_size": imgs[0].shape[1::-1],
        "camera_angle_x": camera_angle_x

    }
    np.savez(cam_outfile, **cam_dict)

    # save out 

    return outfile, scene

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    
    
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01
    niter = 300

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile, trimesh_scene = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                                     clean_depth, transparent_cams, cam_size)
    
    # Instead of displaying the scene using trimesh, we export the scene and camera parameters.
    # Original comment for showing the scene is preserved below (commented out).
    # trimesh_scene.show()

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d/depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs

def main_demo(tmpdirname, model, device, image_size, input_image, silent=False, as_pointcloud=False):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)#, as_pointcloud=as_pointcloud)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    
    # List all files in the input folder
    #input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Filter only image files (assuming common image extensions)
    #image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    input_files = input_image #[f for f in input_files if os.path.splitext(f)[1].lower() in image_extensions]

    # Check if any image files were found
    # if not input_files:
    #     print("No image files found in the specified folder.")
    #     return

    # Check if input_files is empty and if its a picture
    if not input_files:
        raise ValueError("No image files found in the specified folder or input_image is empty.")
    if isinstance(input_files, str):
        input_files = [input_files]
    if not os.path.isfile(input_files[0]):
        raise ValueError(f"The input '{input_files[0]}' is not a valid file. Please provide a valid image file.")

    # Duplicate the input_files list to ensure it contains at least two images
    if len(input_files) == 1:
        input_files = [input_files[0], input_files[0]]

    # Reconstruction options (same as before)
    schedule = 'linear'
    niter = 'niter'
    scenegraph_type = 'complete'
    winsize = 1 
    refid = 1 
    min_conf_thr = 0.3
    cam_size = 1
    # bool to int 
    as_pointcloud = 1 if as_pointcloud else 0
    mask_sky = 0
    clean_depth = 0
    transparent_cams = 1

    scene, outfile, imgs = recon_fun(input_files, schedule, niter, min_conf_thr, as_pointcloud,
                                     mask_sky, clean_depth, transparent_cams, cam_size,
                                     scenegraph_type, winsize, refid)
    print(f"3D model saved to: {outfile}")

if __name__ == '__main__':

        # Load configuration
    parser = argparse.ArgumentParser(description="Run segmentation script with config file.")
    parser.add_argument("--config", default="../src/config.yaml", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)


     # Set up temporary directory
    if config["tmp_dir"] is not None:
        tmp_path = config["tmp_dir"]
    else:
        tmp_path = os.path.join(os.getcwd(), "output")
    os.makedirs(tmp_path, exist_ok=True)
    tempfile.tempdir = tmp_path

    # Set server name
    if config["server_name"] is not None:
        server_name = config["server_name"]
    else:
        server_name = '0.0.0.0' if config["local_network"] else '127.0.0.1'

    # Load model
    if config["weights"] is not None:
        weights_path = config["weights"]
    else:
        weights_path = "naver/" + config["model_name"]
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(config["device"])

    output_dir = config["tmp_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    if not config["silent"]:
        print('Saving outputs in', output_dir)

    # Run the main demo with output_dir specified permanently
    main_demo(
        output_dir, 
        model, 
        config["device"], 
        config["image_size"], 
        config["image_url"], 
        silent=config["silent"], 
        as_pointcloud=config["as_pointcloud"]
        )