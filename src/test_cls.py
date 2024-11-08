import argparse
import numpy as np
import os, glob, shutil, sys
import pandas as pd
import pickle
import random
from scipy import ndimage
import SimpleITK as sitk
from typing import Dict, List, Optional, Tuple
import wandb

#
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    NormalizeIntensity,
)
from monai.utils import set_determinism

from mislight.utils.hydra import instantiate_list


def run_inference(
    input_images,
    weight_keys,
    weight_map_path="./weights/weight_map.pkl",
    cleanup=True,
):
    """
    Args:
        input_images: list of SimpleITK images. assumes [ADC, Z_ADC]
    """

    ## temp dirs
    tmpdir_base = "./tmpdir"
    tmpdir_image = os.path.join(tmpdir_base, "image")
    os.makedirs(tmpdir_image, exist_ok=True)

    ## foreground crop images and save to nifti
    imgarrs = [sitk.GetArrayFromImage(x).transpose([2, 1, 0]) for x in input_images]
    tx1arr = imgarrs[0]

    fg_idx = np.argwhere(tx1arr > 0)
    mx, my, mz = fg_idx.min(0)
    Mx, My, Mz = fg_idx.max(0)
    _cslice = (slice(mx, Mx), slice(my, My), slice(mz, Mz))
    _crop_params = [mx, Mx, my, My, mz, Mz]

    slice_imgs = [x[_cslice] for x in input_images]
    slice_imgarrs = [sitk.GetArrayFromImage(x).transpose([2, 1, 0]) for x in slice_imgs]
    tx1 = slice_imgs[0]
    new_spc = tx1.GetSpacing()
    new_org = tx1.GetOrigin()
    new_dir = tx1.GetDirection()

    re_imgs = []
    for rx in slice_imgarrs:
        re_image1 = sitk.GetImageFromArray(rx.transpose([2, 1, 0]))
        re_image1.SetSpacing(new_spc)
        re_image1.SetOrigin(new_org)
        re_image1.SetDirection(new_dir)
        re_imgs.append(re_image1)

    for i, x in enumerate(re_imgs):
        nxpth = os.path.join(tmpdir_image, f"case_{i:04d}.nii.gz")
        sitk.WriteImage(x, nxpth)

    ### load weights and thrs
    with open(weight_map_path, "rb") as f:
        weight_map = pickle.load(f)

    list_preds = []
    list_preds_bin = []
    list_thrs = []

    for weight_key in weight_keys:
        thr_list = weight_map[weight_key]["thr"]
        run_dir_list = weight_map[weight_key]["run_dir"]
        list_thrs += thr_list

        for _thr, run_dir in zip(thr_list, run_dir_list):
            pretrained_ckpt = glob.glob(os.path.join(run_dir, "checkpoint", "*.ckpt"))[0]
            cfg_override_path = os.path.join(run_dir, "config/overrides.yaml")

            overrides = list(OmegaConf.load(cfg_override_path))
            overrides += [
                "paths.output_dir=temp/logs",
                "data.dataloader.batch_size=1",
                "data.dataloader.batch_size_inference=1",
                "data.dataloader.num_workers=0",
                "train=True",
                "valid=True",
                "seed.deterministic=false",
                f"data.dataset.image_dir={tmpdir_image}",
                "data.dataset.dataset_file=null",
                "+networks.netB.weights=null",
            ]

            with initialize(version_base=None, config_path="../config"):
                cfg = compose(
                    config_name="train",
                    overrides=overrides,
                    return_hydra_config=True,
                )

            if cfg.seed.seed:
                torch.manual_seed(cfg.seed.seed)
            model = instantiate(cfg.model, _recursive_=False)

            model.load_pretrained(pretrained_ckpt)
            model.eval()

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            #device = torch.device(device_type)
            #model.to(device)
            print(f"device_type: {device_type}")
            
            if cfg.seed.seed:
                set_determinism(cfg.seed.seed)
            dm = instantiate(cfg.data, _recursive_=False)

            dm.setup("test")
            dl_test = dm.test_dataloader()

            trainer = instantiate(cfg.trainer, logger=[], callbacks=[])
            ts_preds = trainer.predict(model, dataloaders=dl_test)
            ts_preds = np.array(ts_preds)[:,0]

            """
            ts_preds = []
            ts_metas = {}
            for batch in dl_test:
                for k in batch.keys():
                    try:
                        batch[k] = batch[k].to(device)
                    except:
                        pass

                use_mp = (cfg.trainer.precision == 16)
                #print(use_mp)
                with torch.autocast(device_type, enabled=use_mp, dtype=torch.float16):
                #with torch.autocast(device_type, enabled=use_mp, dtype=torch.bfloat16):
                    with torch.no_grad():
                        outputs = model._step_forward_infer(batch, 0)
                        outputs = model.convert_output(outputs)
                        #print(outputs.dtype)
                        for a in outputs:
                            ts_preds.append(a.float().cpu().numpy())

                        for k, v in batch["metadata"].items():
                            if not k in ts_metas:
                                ts_metas[k] = []
                            ts_metas[k] += v
            """

            tspreds = np.array(ts_preds)

            ts_clabel = (tspreds[:, 1] >= _thr).astype(int)

            list_preds.append(tspreds[0, 1])
            list_preds_bin.append(ts_clabel[0])

    ens_preds_bin = int(np.array(list_preds_bin).sum() > 0.5 * len(list_preds_bin))

    # cleanup
    if cleanup:
        shutil.rmtree(tmpdir_base)

    return ens_preds_bin, list_preds_bin, list_preds, list_thrs


def main():
    # parse the arguments
    parser = argparse.ArgumentParser(
        prog="Inference Script for the BONBID-HIE 2024 Challenge Submission",
        description="Runs inference on an input image and\
                            saves the output to the a folder",
    )

    parser.add_argument("-ii", "--input_image", nargs="+", required=True)
    parser.add_argument("-wk", "--weight_key", required=False)
    parser.add_argument(
        "-wm", "--weight_map", default="./weights/weight_map.pkl", required=False
    )
    parser.add_argument("--no_cleanup", action="store_true")

    args = parser.parse_args()


if __name__ == "__main__":
    main()
