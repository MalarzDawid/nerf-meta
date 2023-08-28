import argparse
import json
import os
import copy
import torch
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from neptune.utils import stringify_unsupported

from dataset import NeRFShapeNetDataset
from load_generalized import load_many_data
from multiplane_nerf_utils import create_mi_nerf, get_rays, render, img2mse, mse2psnr
from multiplane_helpers_generalized import ImagePlane
from multiplane_nerf_utils import device
import numpy as np
from tqdm import tqdm

import neptune
from PIL import Image

from multiplane_nerf_utils import render_path

torch.set_default_device(device)

SCORE_SAMPLES = 5
SAMPLES = 3

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def inner_multiplane_loop(
    inner_model, inner_optim, imgs, poses, hwfk, N_rand, inner_steps
):
    """
    train the inner model for a specified number of iterations
    """
    # Metrics
    inner_psnr, inner_loss = MeanMetric().to(device), MeanMetric().to(device)

    H, W, _, K = hwfk
    img_i = np.random.choice(range(imgs.shape[0]))
    target = imgs[img_i]
    pose = poses[img_i, :3, :4]

    rays_o, rays_d = get_rays(H, W, K, pose)
    coords = torch.stack(
        torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1
    )  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])

    for _ in range(inner_steps):
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        r_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        r_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([r_o, r_d], 0).to(device)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]

        inner_optim.zero_grad()

        rgb, _, _, extras = render(
            H, W, K, rays=batch_rays, verbose=False, retraw=True, **inner_model
        )

        # Calc loss & psnr
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if psnr == torch.inf:
            print("Inv number")
        else:
            inner_psnr.update(psnr)

        inner_loss.update(loss)

        loss.backward()

        inner_optim.step()

    return inner_psnr.compute(), inner_loss.compute()


def train_meta(args, render_kwargs_train, meta_optim, data_loader, hwfk, run):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """

    _, _, focal, _ = hwfk

    # Init metrics
    epoch_loss, epoch_psnr = MeanMetric().to(device), MeanMetric().to(device)

    inner_render_kwargs_train = None

    for batch in tqdm(data_loader):
        imgs, poses = batch["images"][0].float(), batch["cam_poses"][0].float()
        imgs, poses = imgs.to(device), poses.to(device)
        imgs, poses = imgs.squeeze(), poses.squeeze()

        # Reset gradient
        meta_optim.zero_grad()

        # image_plane = ImagePlane(focal, poses.cpu().numpy(), imgs.cpu().numpy(), 50)
        image_plane = ImagePlane(focal, poses, imgs, args.multiplane_views)

        render_kwargs_train["network_fn"].image_plane = image_plane
        render_kwargs_train["network_fine"].image_plane = image_plane

        # Create inner model & optim
        inner_render_kwargs_train = copy.deepcopy(render_kwargs_train)

        grads = list(inner_render_kwargs_train["network_fn"].parameters())
        grads += list(inner_render_kwargs_train["network_fine"].parameters())
        inner_optim = torch.optim.Adam(grads, lr=args.inner_lr)

        psnr, loss = inner_multiplane_loop(
            inner_render_kwargs_train,
            inner_optim,
            imgs,
            poses,
            hwfk,
            args.N_rand,
            args.inner_steps,
        )

        with torch.no_grad():
            for meta_param, inner_param in zip(
                render_kwargs_train["network_fn"].parameters(),
                inner_render_kwargs_train["network_fn"].parameters(),
            ):
                meta_param.grad = meta_param - inner_param
            for meta_param, inner_param in zip(
                render_kwargs_train["network_fine"].parameters(),
                inner_render_kwargs_train["network_fine"].parameters(),
            ):
                meta_param.grad = meta_param - inner_param

        meta_optim.step()

        epoch_psnr.update(psnr)
        epoch_loss.update(loss)

    epoch_psnr_value = epoch_psnr.compute().item()
    epoch_loss_value = epoch_loss.compute().item()

    # Log
    if args.logger:
        run[f"train/psnr"].append(epoch_psnr_value)
        run[f"train/loss"].append(epoch_loss_value)

    print("AVG PSNR: ", epoch_psnr_value)
    print("AVG LOSS: ", epoch_loss_value)


def val_meta(val_model_kwargs, val_dataloader, hwfk, bound, args, run, epoch):
    H, W, focal, K = hwfk
    hwf = torch.Tensor([H, W, focal])

    psnr_tto, loss_tto, psnr_val = [], [], []

    meta_network_fn_trained_state = val_model_kwargs["network_fn"].state_dict()
    meta_network_fine_trained_state = val_model_kwargs["network_fine"].state_dict()
    val_model = copy.deepcopy(val_model_kwargs)

    # Validation step
    for ti, batch in tqdm(enumerate(val_dataloader)):
        imgs, poses = batch["images"][0].float(), batch["cam_poses"][0].float()
        imgs, poses, hwf, bound = (
            imgs.to(device),
            poses.to(device),
            hwf.to(device),
            bound.to(device),
        )
        imgs, poses = imgs.squeeze(), poses.squeeze()

        tto_images, test_imgs = torch.split(
            imgs, [args.tto_views, args.test_views], dim=0
        )
        tto_poses, test_poses = torch.split(
            poses, [args.tto_views, args.test_views], dim=0
        )

        # Create meta optimizer
        val_model["network_fn"].load_state_dict(meta_network_fn_trained_state)
        val_model["network_fine"].load_state_dict(meta_network_fine_trained_state)

        grads = list(val_model["network_fn"].parameters())
        grads += list(val_model["network_fine"].parameters())
        val_optim = torch.optim.Adam(grads, lr=args.inner_lr)

        image_plane = ImagePlane(focal, tto_poses, tto_images, args.multiplane_views)
        val_model["network_fn"].image_plane = image_plane
        val_model["network_fine"].image_plane = image_plane

        tto_psnr, tto_loss = inner_multiplane_loop(
            val_model,
            val_optim,
            tto_images,
            tto_poses,
            hwfk,
            args.N_rand,
            args.tto_steps,
        )

        # Inference
        with torch.no_grad():
            image_plane = ImagePlane(
                focal, test_poses, test_imgs, args.multiplane_views
            )
            val_model["network_fn"].image_plane = image_plane
            val_model["network_fine"].image_plane = image_plane
            rgbs, _, val_psnr = render_path(
                torch.Tensor(test_poses[:SCORE_SAMPLES]),
                [H, W, focal],
                K,
                args.chunk,
                val_model,
                gt_imgs=test_imgs[:SCORE_SAMPLES],
                savedir=None,
            )  # images

            if ti < SAMPLES:
                for i in range(SAMPLES):
                    rgb8 = to8b(rgbs[i])
                    gt = to8b(test_imgs[i].cpu().numpy())
                    output_img = np.hstack((rgb8, gt))
                    img = Image.fromarray(output_img)
                    if args.logger:
                        run[f"images/{ti}_{i}"].append(img, step=epoch)

        psnr_val.append(val_psnr)
        psnr_tto.append(tto_psnr)
        loss_tto.append(tto_loss)

    epoch_psnr_tto_float = torch.Tensor(psnr_tto).mean().item()
    epoch_loss_tto_float = torch.Tensor(loss_tto).mean().item()
    epoch_psnr_val_float = torch.Tensor(psnr_val).mean().item()

    print("VAL AVG PSNR: ", epoch_psnr_tto_float)
    print("VAL AVG LOSS: ", epoch_loss_tto_float)

    if args.logger:
        run[f"val/psnr"].append(epoch_psnr_val_float)
        run[f"tto/psnr"].append(epoch_psnr_tto_float)
        run[f"tto/loss"].append(epoch_loss_tto_float)
    return epoch_psnr_val_float


def train(args):
    # Logger
    run = None
    # Create a Neptune run object
    if args.logger:
        run = neptune.init_run(
            project=args.neptune_project,
            tags=args.neptune_tags,  # optional
        )
        run["parameters"] = stringify_unsupported(args)

    # Create train dataset
    train_set = NeRFShapeNetDataset(root_dir=args.root_dir, classes=[args.class_name])
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, generator=torch.Generator(device=device)
    )

    # Create validation dataset
    val_set = NeRFShapeNetDataset(
        root_dir=args.root_dir, classes=[args.class_name], train=False
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, generator=torch.Generator(device=device)
    )

    # objects, test_objects, render_poses, hwf
    _, _, _, hwf = load_many_data(
        str(os.path.join(args.root_dir, args.class_name))
    )  # TODO

    # Prepare focal and K
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = torch.tensor([H, W, focal])

    bds_dict = {
        "near": float(args.bound_near),
        "far": float(args.bound_far),
    }

    bound = torch.tensor([v for v in bds_dict.values()])

    K = np.array([[focal.item(), 0, 0.5 * W], [0, focal.item(), 0.5 * H], [0, 0, 1]])
    hwfk = [H, W, focal, K]

    # Get meta_model and meta_optim
    meta_model_kwargs, _, _, _, meta_optim = create_mi_nerf(args.multiplane_views, args)
    meta_model_kwargs.update(bds_dict)

    psnr_score = []
    for epoch in range(1, args.meta_epochs+1):
        # Train meta model
        print("*" * 50)
        print("Epoch: ", epoch)
        print("*" * 50)
        train_meta(args, meta_model_kwargs, meta_optim, train_loader, hwfk, run)

        # TTO Optimization
        if epoch % 5 == 0 or epoch == 1:
            print("*" * 50)
            print("VALIDATION\n")
            psnr = val_meta(
                meta_model_kwargs, val_loader, hwfk, bound, args, run, epoch
            )
            psnr_score.append(psnr)

        # Save model after epoch
        torch.save(
            {
                "epoch": epoch,
                "meta_model_fn_state_dict": meta_model_kwargs[
                    "network_fn"
                ].state_dict(),
                "meta_model_fine_state_dict": meta_model_kwargs[
                    "network_fine"
                ].state_dict(),
                "meta_optim_state_dict": meta_optim.state_dict(),
            },
            f"tmp/meta_epoch{epoch}.pth",
        )
        if epoch % 5 == 0:
            run[f"meta_model/epoch_{epoch}"].upload(f"tmp/meta_epoch{epoch}.pth")

    if args.logger:
        run.stop()

    return sum(psnr_score) / len(psnr_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shapenet few-shot view synthesis")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config file for the shape class (cars, chairs or lamps)",
    )
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value
    score = train(args)
