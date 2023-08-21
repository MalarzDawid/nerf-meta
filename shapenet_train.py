import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.rendering import get_rays_shapenet, sample_points, volume_render
from dataset import NeRFShapeNetDataset
from load_generalized import load_many_data
from multiplane_nerf_utils import create_mi_nerf, get_rays, render, img2mse, mse2psnr
from multiplane_helpers_generalized import ImagePlane
from multiplane_nerf_utils import device
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import neptune
from PIL import Image
import imageio
from multiplane_nerf_utils import render_path

torch.set_default_device(device)

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)



def inner_multiplane_loop(inner_model, inner_optim, imgs, poses, hwfk, N_rand, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    # Metrics
    local_psnr, local_loss = [], []

    H, W, _, K = hwfk
    img_i = np.random.choice(range(imgs.shape[0]))
    target = imgs[img_i]
    pose = poses[img_i, :3, :4]

    rays_o, rays_d = get_rays(H, W, K, pose)
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])
    
    step_size = inner_steps / 2

    for step in range(inner_steps):
        if step %  step_size == 0:
            print("Step: ", step)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        r_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        r_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([r_o, r_d], 0).to(device)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]

        inner_optim.zero_grad()

        rgb, disp, acc, extras = render(H, W, K, rays=batch_rays,
                                        verbose=False, retraw=True,
                                        **inner_model)

        # Calc loss & psnr
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        local_psnr.append(psnr.item())
        local_loss.append(loss.item())

        loss.backward()

        inner_optim.step()

    local_psnr = torch.Tensor(local_psnr).mean().item()
    local_loss = torch.Tensor(local_loss).mean().item()

    return local_psnr, local_loss


def train_meta(args, render_kwargs_train, meta_optim, data_loader, hwfk, run):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """

    _, _, focal, _ = hwfk
    # hwf = torch.Tensor([H, W, focal]).to(device)

    epoch_psnr = []
    epoch_loss = []
    inner_render_kwargs_train = None
    for batch in tqdm(data_loader):
        imgs, poses = batch["images"][0].float(), batch["cam_poses"][0].float()
        imgs, poses = imgs.to(device), poses.to(device)
        imgs, poses = imgs.squeeze(), poses.squeeze()

        # Reset gradient
        meta_optim.zero_grad()

        # image_plane = ImagePlane(focal, poses.cpu().numpy(), imgs.cpu().numpy(), 50)
        image_plane = ImagePlane(focal, poses, imgs, args.multiplane_views)


        render_kwargs_train['network_fn'].image_plane = image_plane
        render_kwargs_train['network_fine'].image_plane = image_plane

        # Create inner model & optim
        inner_render_kwargs_train = copy.deepcopy(render_kwargs_train)

        grads = list(inner_render_kwargs_train["network_fn"].parameters())
        grads += list(inner_render_kwargs_train["network_fine"].parameters())
        inner_optim = torch.optim.Adam(grads, lr=args.inner_lr)

        psnr, loss = inner_multiplane_loop(inner_render_kwargs_train, inner_optim, imgs, poses, hwfk, args.N_rand,
                              args.inner_steps)

        with torch.no_grad():
            for meta_param, inner_param in zip(render_kwargs_train["network_fn"].parameters(), inner_render_kwargs_train["network_fn"].parameters()):
                meta_param.grad = meta_param - inner_param
            for meta_param, inner_param in zip(render_kwargs_train["network_fine"].parameters(), inner_render_kwargs_train["network_fine"].parameters()):
                meta_param.grad = meta_param - inner_param
        
        meta_optim.step()
        epoch_psnr.append(psnr)
        epoch_loss.append(loss)

    epoch_psnr_float = torch.Tensor(epoch_psnr).mean().item()
    epoch_loss_float = torch.Tensor(epoch_loss).mean().item()
    run[f"train/psnr"].append(epoch_psnr_float)
    run[f"train/loss"].append(epoch_loss_float)

    print("AVG PSNR: ", epoch_psnr_float)
    print("AVG LOSS: ", epoch_loss_float)
    return None, None, render_kwargs_train


def main():
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

    # Logger
    run = None
    # Create a Neptune run object
    run = neptune.init_run(
        project=args.neptune_project,
        tags=args.neptune_tags,  # optional
    )

    # Create train dataset
    train_set = NeRFShapeNetDataset(root_dir="data/multiple", classes=["cars"])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, generator=torch.Generator(device=device))

    # Create validation dataset
    val_set = NeRFShapeNetDataset(root_dir="data/multiple", classes=["cars"], train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, generator=torch.Generator(device=device))

    # objects, test_objects, render_poses, hwf
    _, _, _, hwf = load_many_data(f'data/multiple/cars')

    # Prepare focal and K
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = torch.tensor([H, W, focal])

    # TODO export bound to config file
    bds_dict = {
        'near': 2.,
        'far': 6.,
    }

    bound = torch.tensor([v for v in bds_dict.values()]) 

    K = np.array([
        [focal.item(), 0, 0.5 * W],
        [0, focal.item(), 0.5 * H],
        [0, 0, 1]
    ])
    hwfk = [H, W, focal, K]

    # Get meta_model and meta_optim
    meta_model_kwargs, _, _, _, meta_optim = create_mi_nerf(args.multiplane_views, args)
    meta_model_kwargs.update(bds_dict)
    
    for epoch in range(1, args.meta_epochs+1):
        _, _, _ = train_meta(args, meta_model_kwargs, meta_optim, train_loader, hwfk, run)

        print("*"*50)
        print("VALIDATION\n")
        # TTO Optimization
        _ = val_meta(meta_model_kwargs, val_loader, hwfk, bound, args, run, epoch)
        
        # Save model after epoch
        torch.save({
            'epoch': epoch,
            'meta_model_fn_state_dict': meta_model_kwargs["network_fn"].state_dict(),
            'meta_model_fine_state_dict': meta_model_kwargs["network_fine"].state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            }, f'o/meta_epoch{epoch}.pth')
    run.stop()


def val_meta(val_model_kwargs, val_dataloader, hwfk, bound, args, run, epoch):
    H, W, focal, K = hwfk
    hwf = torch.Tensor([H, W, focal])

    psnr_val, loss_val = [], []

    meta_network_fn_trained_state = val_model_kwargs["network_fn"].state_dict()
    meta_network_fine_trained_state = val_model_kwargs["network_fine"].state_dict()
    val_model = copy.deepcopy(val_model_kwargs)

    # Validation step
    for ti, batch in tqdm(enumerate(val_dataloader)):
        print(f"\nSample: #{ti}")
        imgs, poses = batch["images"][0].float(), batch["cam_poses"][0].float()
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses = imgs.squeeze(), poses.squeeze()

        tto_images, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        # Create meta optimizer
        val_model["network_fn"].load_state_dict(meta_network_fn_trained_state)
        val_model["network_fine"].load_state_dict(meta_network_fine_trained_state)
        
        grads = list(val_model["network_fn"].parameters())
        grads += list(val_model["network_fine"].parameters())
        val_optim = torch.optim.Adam(grads, lr=args.inner_lr)

        image_plane = ImagePlane(focal, tto_poses, tto_images, args.multiplane_views)
        val_model['network_fn'].image_plane = image_plane
        val_model['network_fine'].image_plane = image_plane
        
        psnr, loss = inner_multiplane_loop(val_model, val_optim, tto_images, tto_poses, hwfk, args.N_rand, args.tto_steps)
        
        # Inference
        with torch.no_grad():
            SAMPLES = 3
            image_plane = ImagePlane(focal, test_poses, test_imgs, args.multiplane_views)
            val_model['network_fn'].image_plane = image_plane
            val_model['network_fine'].image_plane = image_plane
            rgbs, _, _ = render_path(torch.Tensor(test_poses[:SAMPLES]), [H, W, focal], K, args.chunk, val_model,
                                  gt_imgs=test_imgs[:SAMPLES], savedir=None)  # images

            for i in range(SAMPLES):
                rgb8 = to8b(rgbs[i])
                gt = to8b(test_imgs[i].cpu().numpy())
                output_img = np.hstack((rgb8, gt))
                img = Image.fromarray(output_img)
                filename = os.path.join("images", "testset", f"img_{ti}_{i}.png")
                run[str(filename)].append(img, step=epoch)
        
        psnr_val.append(psnr)
        loss_val.append(loss)

    epoch_psnr_float = torch.Tensor(psnr_val).mean().item()
    epoch_loss_float = torch.Tensor(loss_val).mean().item()

    run[f"val/psnr"].append(epoch_psnr_float)
    run[f"val/loss"].append(epoch_loss_float)

    print("VAL AVG PSNR: ", epoch_psnr_float)
    print("VAL AVG LOSS: ", epoch_loss_float)


if __name__ == '__main__':
    main()