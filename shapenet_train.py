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

import imageio
from multiplane_nerf_utils import render_path

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def inner_multiplane_loop(inner_model, inner_optim, imgs, poses, hwfk, N_rand, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    H, W, _, K = hwfk
    img_i = np.random.choice(range(imgs.shape[0]))
    target = imgs[img_i]
    pose = poses[img_i, :3, :4]

    # Remove
    target = torch.Tensor(target).to(device)
    pose = pose.to("cpu")

    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])

    total_pnsr, total_loss = [], []

    for step in range(inner_steps):
        if step %  100 == 0:
            print("Step: ", step)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        r_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        r_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([r_o, r_d], 0)
        batch_rays = batch_rays.to("cpu")
        target_s = target[select_coords[:, 0], select_coords[:, 1]]

        network = "network_fine"
        # for param in inner_model[network].parameters():
        #     print("Before", param.grad)
        #     break
        inner_optim.zero_grad()
        # for param in inner_model[network].parameters():
        #     print("After", param.grad)
        #     break

        rgb, disp, acc, extras = render(H, W, K, rays=batch_rays,
                                        verbose=False, retraw=True,
                                        **inner_model)

        # Remove
        rgb = rgb.to("cpu")
        target_s = target_s.to("cpu")

        # Calc loss & psnr
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(loss.to("cpu"))

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # total_loss = total_loss + loss.item()

        total_pnsr.append(float(psnr))
        total_loss.append(float(loss.item()))

        loss.backward()

        inner_optim.step()

    return total_pnsr, total_loss


def train_meta(args, render_kwargs_train, meta_optim, data_loader, hwfk, device):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """

    H, W, focal, K = hwfk
    hwf = torch.Tensor([H, W, focal])

    epoch_psnr = []
    epoch_loss = []

    for batch_idx, batch in tqdm(enumerate(data_loader)):
        imgs, poses = batch["images"][0].float(), batch["cam_poses"][0].float()
        imgs, poses, hwf = imgs.to(device), poses.to(device), hwf.to(device)
        imgs, poses = imgs.squeeze(), poses.squeeze()

        # Reset gradient
        meta_optim.zero_grad()

        image_plane = ImagePlane(focal, poses.cpu().numpy(), imgs.cpu().numpy(), 50)

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
                # print(f"Meta param: {meta_param} Inner param: {inner_param} New: {meta_param-inner_param}")
                meta_param.grad = meta_param - inner_param
            for meta_param, inner_param in zip(render_kwargs_train["network_fine"].parameters(), inner_render_kwargs_train["network_fine"].parameters()):
                # print(f"Meta param: {meta_param} Inner param: {inner_param} New: {meta_param-inner_param}")
                meta_param.grad = meta_param - inner_param
        
        meta_optim.step()
        epoch_psnr.append(sum(psnr)/args.inner_steps)
        epoch_loss.append(sum(loss) / args.inner_steps)

    print("AVG PSNR: ", sum(epoch_psnr)/len(epoch_psnr))
    print("AVG LOSS: ", sum(epoch_loss)/len(epoch_psnr))
    return epoch_psnr, epoch_loss


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

    # Create train dataset
    train_set = NeRFShapeNetDataset(root_dir="data/multiple", classes=["cars"])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    # Create validation dataset
    val_set = NeRFShapeNetDataset(root_dir="data/multiple", classes=["cars"])
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

    objects, test_objects, render_poses, hwf = load_many_data(f'data/multiple/cars')

    # Prepare focal and K
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = torch.tensor([H, W, focal])
    bound = torch.tensor([2., 6.])

    K = np.array([
        [focal.item(), 0, 0.5 * W],
        [0, focal.item(), 0.5 * H],
        [0, 0, 1]
    ])
    hwfk = [H, W, focal, K]

    # Model init
    meta_render_kwargs_train, render_kwargs_test, start, grad_vars, _ = create_mi_nerf(50, args)

    bds_dict = {
        'near': 2.,
        'far': 6.,
    }
    meta_render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Create meta optimizer
    meta_optim = torch.optim.Adam(grad_vars, args.inner_lr)

    losses = []
    psnrs = []

    for epoch in range(1, args.meta_epochs+1):
        print("*"*50, f"Epoch: {epoch} TRAIN")
        psnr, loss = train_meta(args, meta_render_kwargs_train, meta_optim, train_loader, hwfk, device)
        losses.append(loss)
        psnrs.append(psnr)

        # Validation step
        for ti, batch in enumerate(val_loader):
            if ti >= 5:
                break
            imgs, poses = batch["images"][0].float(), batch["cam_poses"][0].float()
            imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
            imgs, poses = imgs.squeeze(), poses.squeeze()

            # Prepare savedir
            testsavedir = os.path.join("output", "cars", f'testset_{epoch}_{ti}', f'{ti}')
            os.makedirs(testsavedir, exist_ok=True)

            # Inference
            with torch.no_grad():
                image_plane = ImagePlane(focal, poses.cpu().numpy(), imgs.cpu().numpy(), 50)
                render_kwargs_test['network_fn'].image_plane = image_plane
                render_kwargs_test['network_fine'].image_plane = image_plane
                _, _, p = render_path(torch.Tensor(poses[0:5]).cpu(), [H, W, focal], K, args.chunk, render_kwargs_test,
                                      gt_imgs=imgs[0:5], savedir=testsavedir)  # images
                imageio.imwrite(os.path.join(testsavedir, f'gt.png'), to8b(imgs[0].cpu().numpy()))

        # Save model after epoch
        torch.save({
            'epoch': epoch,
            'meta_model_fn_state_dict': meta_render_kwargs_train["network_fn"].state_dict(),
            'meta_model_fine_state_dict': meta_render_kwargs_train["network_fine"].state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            }, f'outputs/meta_epoch{epoch}.pth')
    plt.plot(losses)
    plt.savefig(f"loss.png")
    plt.close()

if __name__ == '__main__':
    main()