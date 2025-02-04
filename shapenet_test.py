from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render
from dataset import NeRFShapeNetDataset
from load_generalized import load_many_data
from multiplane_nerf_utils import create_mi_nerf
import numpy as np
from multiplane_nerf_utils import device, get_rays, render, img2mse, render_path
from multiplane_helpers_generalized import ImagePlane


def test_time_optimize_multiplane(args, optim, imgs, poses, hwk, render_kwargs_train):
    """
    test-time-optimize the meta trained model on available views
    """
    H, W, K = hwk

    img_i = np.random.choice(list(range(imgs.shape[0])))
    target = imgs[img_i]
    target = torch.Tensor(target).to(device)
    pose = poses[img_i, :3, :4]
    pose = pose.to("cpu")

    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])

    for step in range(args.tto_steps):
        print("Step: ", step)
        select_inds = np.random.choice(coords.shape[0], size=[args.tto_batchsize], replace=False)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        r_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        r_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([r_o, r_d], 0)
        batch_rays = batch_rays.to("cpu")
        target_s = target[select_coords[:, 0], select_coords[:, 1]]

        optim.zero_grad()
        rgb, disp, acc, extras = render(H, W, K, chunk=512*32, rays=batch_rays,
                                        verbose=False, retraw=True,
                                        **render_kwargs_train)
        rgb = rgb.to(device)
        target_s = target_s.to(device)
        img_loss = img2mse(rgb, target_s)
        loss = img_loss

        loss.backward()
        optim.step()


def test_time_optimize(args, model, optim, imgs, poses, hwf, bound):
    """
    test-time-optimize the meta trained model on available views
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(args.tto_steps):
        indices = torch.randint(num_rays, size=[args.tto_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()


def report_result(args, model, imgs, poses, hwf, bound):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    args.num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, args.test_batchsize):
                rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
                color_batch = volume_render(rgbs_batch, sigmas_batch,
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def test():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                    help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=-1,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    parser.add_argument("--dataset_sample", type=str, default='cars',
                        help='')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')
    
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = NeRFShapeNetDataset(root_dir="data/multiple", classes=["cars"], train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    objects, test_objects, render_poses, hwf = load_many_data(f'data/multiple/cars')

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_mi_nerf(50, args)
    global_step = start

    bds_dict = {
        'near': 2.,
        'far': 6.,
    }
    render_kwargs_train.update(bds_dict)

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True)
    
    test_psnrs = []

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf_n = [H, W, focal]
    hwf = torch.tensor([H, W, focal])
    bound = torch.tensor([2., 6.])

    K = np.array([
        [focal.item(), 0, 0.5 * W],
        [0, focal.item(), 0.5 * H],
        [0, 0, 1]
    ])
    hwk = (H, W, K)

    for idx, (batch) in enumerate(test_loader):
        imgs, poses = batch["images"][0].float(), batch["cam_poses"][0].float()
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses = imgs.squeeze(), poses.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        optim = torch.optim.Adam(render_kwargs_train["network_fn"].parameters(), args.tto_lr)

        image_plane = ImagePlane(focal, poses.cpu().numpy(), imgs.cpu().numpy(), 50)

        render_kwargs_train['network_fn'].image_plane = image_plane
        render_kwargs_train['network_fine'].image_plane = image_plane

        print(f"Test {idx}")
        test_time_optimize_multiplane(args, optim, tto_imgs, tto_poses, hwk, render_kwargs_train)
        with torch.no_grad():
            render_path(test_poses.to("cpu"), hwf_n, K, args.chunk, render_kwargs_train,
                        gt_imgs=test_imgs.to("cpu"), savedir="o")
        # test_time_optimize(args, render_kwargs_train, optim, tto_imgs, tto_poses, hwf, bound)
        # scene_psnr = report_result(args, model, test_imgs, test_poses, hwf, bound)
        #
        # create_360_video(args, model, hwf, bound, device, idx+1, savedir)

        # print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        # test_psnrs.append(scene_psnr)
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")


if __name__ == '__main__':
    test()