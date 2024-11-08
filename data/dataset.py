import copy
import glob
import importlib
import itertools
import json
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import random
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from monai.transforms import Compose, LoadImage, BorderPad

from mislight.utils.hydra import instantiate_list

from .utils import TwoStreamSampler

class BONBIDSegmentationDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        label_dir=None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        image_path = self.image_paths[index % self.image_size]
        read_items['image'] = image_path
        metadata['image_path'] = image_path 
        
        if hasattr(self, "label_paths"):
            label_path = self.label_paths[index % self.image_size]
            read_items["label"] = label_path
            # read_items['label_raw'] = label_path
            metadata["label_path"] = label_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        image_extension="nii.gz",
        label_extension="nii.gz",
        select_channels:Sequence[int]=None,
        **kwargs,
    ):
        all_keys = []
        _paths = sorted(glob.glob(os.path.join(self.image_dir, f"*.{image_extension}")))
        _keys = set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths)
        all_keys.append(_keys)
        if getattr(self, "label_dir", None) is not None:
            _paths = sorted(glob.glob(os.path.join(self.label_dir, f"*.{image_extension}")))
            _keys = set(os.path.basename(x).split(f".{label_extension}")[0] for x in _paths)
            all_keys.append(_keys)
        _c_keys = sorted(set.intersection(*all_keys))
        
        if dataset_file is None:
            if self.phase in ["train", "valid"]:
                kf = KFold(n_splits=cv_split, shuffle=True, random_state=split_seed)
                _filtered_idx = (
                    [x for x, _ in kf.split(_c_keys)][cv_fold]
                    if self.phase == "train"
                    else [x for _, x in kf.split(_c_keys)][cv_fold]
                )
                _filtered_keys = [_c_keys[i] for i in _filtered_idx]
        else:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _c_keys if x in this_split]

        if select_channels is None:
            select_channels = [0]
        image_paths = []
        for k in _filtered_keys:
            _ips = [os.path.join(self.image_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
            if all([os.path.exists(x) for x in _ips]):
                image_paths.append(_ips)
        self.image_paths = image_paths
        self.image_size = len(image_paths)
        print_str = f"image num: {self.image_size}"

        if getattr(self, "label_dir", None) is not None:
            _paths = [os.path.join(self.label_dir, f"{k}.{label_extension}") for k in _filtered_keys]
            _paths = [x for x in _paths if os.path.exists(x)]
            self.label_paths = _paths
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        print(print_str)
        
        
class BONBIDClassificationDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        label_dir=None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.clabel_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        image_path = self.image_paths[index % self.image_size]
        read_items['image'] = image_path
        metadata['image_path'] = image_path 
        
        if hasattr(self, "clabel_paths"):
            clabel_path = self.clabel_paths[index % self.image_size]
            read_items["clabel"] = clabel_path
            # read_items['label_raw'] = label_path
            metadata["clabel_path"] = clabel_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        table_file,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        image_extension="nii.gz",
        label_extension="nii.gz",
        select_channels:Sequence[int]=None,
        **kwargs,
    ):
        with open(table_file, "rb") as f:
            dftb = pickle.load(f)
        
        all_keys = [set(dftb.keys())]
        _paths = sorted(glob.glob(os.path.join(self.image_dir, f"*.{image_extension}")))
        _keys = set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths)
        all_keys.append(_keys)
        if getattr(self, "label_dir", None) is not None:
            _paths = sorted(glob.glob(os.path.join(self.label_dir, f"*.{image_extension}")))
            _keys = set(os.path.basename(x).split(f".{label_extension}")[0] for x in _paths)
            all_keys.append(_keys)
        _c_keys = sorted(set.intersection(*all_keys))
        
        if dataset_file is None:
            if self.phase in ["train", "valid"]:
                kf = KFold(n_splits=cv_split, shuffle=True, random_state=split_seed)
                _filtered_idx = (
                    [x for x, _ in kf.split(_c_keys)][cv_fold]
                    if self.phase == "train"
                    else [x for _, x in kf.split(_c_keys)][cv_fold]
                )
                _filtered_keys = [_c_keys[i] for i in _filtered_idx]
        else:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _c_keys if x in this_split]

        if select_channels is None:
            select_channels = [0]
        image_paths = []
        for k in _filtered_keys:
            _ips = [os.path.join(self.image_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
            if all([os.path.exists(x) for x in _ips]):
                image_paths.append(_ips)
        self.image_paths = image_paths
        self.image_size = len(image_paths)
        print_str = f"image num: {self.image_size}"

        if getattr(self, "label_dir", None) is not None:
            _paths = [os.path.join(self.label_dir, f"{k}.{label_extension}") for k in _filtered_keys]
            _paths = [x for x in _paths if os.path.exists(x)]
            self.label_paths = _paths
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        _paths = [dftb[k] for k in _filtered_keys]            
        self.clabel_paths = np.array(_paths)
        self.clabel_size = len(_paths)
        print_str += f", clabel num: {self.clabel_size}"

        print(print_str)
        
class BONBIDClassificationDatasetV2(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        label_dir=None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.clabel_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        image_path = self.image_paths[index % self.image_size]
        read_items['image'] = image_path
        metadata['image_path'] = image_path 
        
        if hasattr(self, "clabel"):
            clabel_x = self.clabel[index % self.image_size]
            read_items["clabel"] = clabel_x
            metadata["clabel"] = clabel_x

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        image_extension="nii.gz",
        label_extension="nii.gz",
        select_channels:Sequence[int]=None,
        override_split_phase=None,
        **kwargs,
    ):
        NO_LABEL = -1
        NO_CLABEL = -1  
        if select_channels is None:
            select_channels = [0]
        if override_split_phase is None:
            ppp = self.phase
        else:
            ppp = override_split_phase
        
        if dataset_file is not None:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
            DATA_CLABEL = dsf['clabel']
            if 'clabel_pseudo' in dsf:
                DATA_CLABEL_PSEUDO = dsf['clabel_pseudo']
            else:
                DATA_CLABEL_PSEUDO = {}
            
        if self.phase in ['train', 'valid']:
            if self.phase == 'train':
                z = DATA_CLABEL.copy()
                z.update(DATA_CLABEL_PSEUDO)
                DATA_CLABEL = z
            
            _dir = self.image_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            this_split = dsf["split"][cv_fold][ppp]
            _filtered_keys = [x for x in _keys if (x in this_split) and (x in DATA_CLABEL.keys())]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.image_size = _size
            self.image_paths = image_paths
            self.image_keys = _filtered_keys
            
            self.clabel = np.array([DATA_CLABEL[x] for x in _filtered_keys])
            self.clabel_size = len(self.clabel)
            count_clabel = sum([x!=NO_CLABEL for x in self.clabel])
            print_str = f"image num: {self.image_size} (clabel {count_clabel})"
            
        elif self.phase == 'valid_test':
            _dir = self.image_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            if override_split_phase is None:
                ppp = 'valid'
            else:
                ppp = override_split_phase
            
            this_split = dsf["split"][cv_fold][ppp]
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.image_size = _size
            self.image_paths = image_paths
            self.image_keys = _filtered_keys

            print_str = f"image num: {self.image_size})"
            
        elif self.phase == 'test':
            _dir = self.image_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            _filtered_keys = _keys
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.image_size = _size
            self.image_paths = image_paths
            self.image_keys = _filtered_keys

            print_str = f"image num: {self.image_size}"
        
        else:
            raise ValueError(f'{self.phase} is not recognized')  
        
        print(print_str)        


class BONBIDSegClsDatasetSSL(Dataset):
    def __init__(
        self,
        transform,
        imageA_dir,
        labelA_dir=None,
        imageB_dir=None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.imageA_dir = imageA_dir
        self.labelA_dir = labelA_dir
        self.imageB_dir = imageB_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.imageA_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.imageA_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        imageA_path = self.imageA_paths[index % self.imageA_size]
        read_items['imageA'] = imageA_path
        metadata['imageA_path'] = imageA_path 
        
        if hasattr(self, "labelA_paths"):
            labelA_path = self.labelA_paths[index % self.imageA_size]
            read_items["labelA"] = labelA_path
            metadata["labelA_path"] = labelA_path
        if hasattr(self, 'clabelA'):
            clabelA = self.clabelA[index % self.imageA_size]
            read_items['clabelA'] = clabelA
            metadata['clabelA'] = clabelA
            
        if hasattr(self, "imageB_paths"):
            index_B = random.randint(0, self.imageB_size - 1)
            imageB_path = self.imageB_paths[index_B % self.imageB_size]
            read_items['imageB'] = imageB_path
            metadata['imageB_path'] = imageB_path
            if hasattr(self, 'clabelB'):
                clabelB = self.clabelB[index_B % self.imageB_size]
                read_items['clabelB'] = clabelB
                metadata['clabelB'] = clabelB

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        image_extension="nii.gz",
        label_extension="nii.gz",
        select_channels:Sequence[int]=None,
        override_split_phase=None,
        **kwargs,
    ):
        with open(dataset_file, "rb") as f:
            dsf = pickle.load(f)
        NO_LABEL = -1
        NO_CLABEL = -1
        DATA_CLABEL = dsf['clabel']
        if select_channels is None:
            select_channels = [0]
        
        if self.phase == 'train':
            _dir = self.imageA_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.imageA_size = _size
            self.imageA_paths = image_paths
            self.imageA_keys = _filtered_keys
            
            self.labelA_paths = [os.path.join(self.labelA_dir, f'{x}.{label_extension}') for x in _filtered_keys]
            self.labelA_size = len(self.labelA_paths)
            self.clabelA = np.array([DATA_CLABEL[x] if x in DATA_CLABEL.keys() else NO_CLABEL for x in _filtered_keys])
            count_clabel = sum([x!=NO_CLABEL for x in self.clabelA])
            print_str = f"imageA num: {self.imageA_size} (clabel {count_clabel})"
            
            _dir = self.imageB_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.imageB_size = _size
            self.imageB_paths = image_paths
            self.imageB_keys = _filtered_keys
            self.clabelB = np.array([DATA_CLABEL[x] if x in DATA_CLABEL.keys() else NO_CLABEL for x in _filtered_keys])
            count_clabel = sum([x!=NO_CLABEL for x in self.clabelB])
            print_str += f", imageB num: {self.imageB_size} (clabel {count_clabel})"
            
        elif self.phase == 'valid':
            _dir = self.imageA_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            if override_split_phase is None:
                ppp = self.phase
            else:
                ppp = override_split_phase
            this_split = dsf["split"][cv_fold][ppp]
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.imageA_size = _size
            self.imageA_paths = image_paths
            self.imageA_keys = _filtered_keys
            
            self.labelA_paths = [os.path.join(self.labelA_dir, f'{x}.{label_extension}') for x in _filtered_keys]
            self.labelA_size = len(self.labelA_paths)
            self.clabelA = [DATA_CLABEL[x] if x in DATA_CLABEL.keys() else NO_CLABEL for x in _filtered_keys]
            count_clabel = sum([x!=NO_CLABEL for x in self.clabelA])
            print_str = f"imageA num: {self.imageA_size} (clabel {count_clabel})"
        
        elif self.phase == 'valid_test':
            _dir = self.imageA_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            this_split = dsf["split"][cv_fold]['valid']
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.imageA_size = _size
            self.imageA_paths = image_paths
            self.imageA_keys = _filtered_keys
            
            self.clabelA = [DATA_CLABEL[x] if x in DATA_CLABEL.keys() else NO_CLABEL for x in _filtered_keys]
            count_clabel = sum([x!=NO_CLABEL for x in self.clabelA])
            print_str = f"imageA num: {self.imageA_size} (clabel {count_clabel})"
            
        else:
            raise ValueError(f'{self.phase} is not recognized')  
        
        print(print_str)


class BONBIDSegClsDatasetFSL(BONBIDSegClsDatasetSSL):
    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        image_extension="nii.gz",
        label_extension="nii.gz",
        select_channels:Sequence[int]=None,
        override_split_phase=None,
        **kwargs,
    ):
        with open(dataset_file, "rb") as f:
            dsf = pickle.load(f)
        NO_LABEL = -1
        NO_CLABEL = -1
        DATA_CLABEL = dsf['clabel']
        if select_channels is None:
            select_channels = [0]
        
        if self.phase == 'train':
            _dir = self.imageA_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.imageA_size = _size
            self.imageA_paths = image_paths
            self.imageA_keys = _filtered_keys
            
            self.labelA_paths = [os.path.join(self.labelA_dir, f'{x}.{label_extension}') for x in _filtered_keys]
            self.labelA_size = len(self.labelA_paths)
            self.clabelA = np.array([DATA_CLABEL[x] if x in DATA_CLABEL.keys() else NO_CLABEL for x in _filtered_keys])
            count_clabel = sum([x!=NO_CLABEL for x in self.clabelA])
            print_str = f"imageA num: {self.imageA_size} (clabel {count_clabel})"
            
        elif self.phase == 'valid':
            _dir = self.imageA_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            if override_split_phase is None:
                ppp = self.phase
            else:
                ppp = override_split_phase
            this_split = dsf["split"][cv_fold][ppp]
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.imageA_size = _size
            self.imageA_paths = image_paths
            self.imageA_keys = _filtered_keys
            
            self.labelA_paths = [os.path.join(self.labelA_dir, f'{x}.{label_extension}') for x in _filtered_keys]
            self.labelA_size = len(self.labelA_paths)
            self.clabelA = [DATA_CLABEL[x] if x in DATA_CLABEL.keys() else NO_CLABEL for x in _filtered_keys]
            count_clabel = sum([x!=NO_CLABEL for x in self.clabelA])
            print_str = f"imageA num: {self.imageA_size} (clabel {count_clabel})"
        
        elif self.phase == 'valid_test':
            _dir = self.imageA_dir
            _paths = sorted(glob.glob(os.path.join(_dir, f"*.{image_extension}")))
            _keys = sorted(set('_'.join(os.path.basename(x).split('_')[:-1]) for x in _paths))
            
            this_split = dsf["split"][cv_fold]['valid']
            _filtered_keys = [x for x in _keys if x in this_split]
            _size = len(_filtered_keys)
                        
            image_paths = []
            for k in _filtered_keys:
                _ips = [os.path.join(_dir, f"{k}_{i:04d}.{image_extension}") for i in select_channels]
                if all([os.path.exists(x) for x in _ips]):
                    image_paths.append(_ips)
            
            self.imageA_size = _size
            self.imageA_paths = image_paths
            self.imageA_keys = _filtered_keys
            
            self.clabelA = [DATA_CLABEL[x] if x in DATA_CLABEL.keys() else NO_CLABEL for x in _filtered_keys]
            count_clabel = sum([x!=NO_CLABEL for x in self.clabelA])
            print_str = f"imageA num: {self.imageA_size} (clabel {count_clabel})"
            
        else:
            raise ValueError(f'{self.phase} is not recognized')  
        
        print(print_str)



















class BMDXRDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        label_name: str = None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.label_name = label_name
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        for k, p in self.image_paths.items():
            imageX_path = p[index % self.image_size]
            read_items[k] = imageX_path
            metadata[f"{k}_path"] = imageX_path

        if hasattr(self, "table_paths"):
            table_path = self.table_paths[index % self.image_size]
            read_items["table"] = table_path

        if hasattr(self, "label_paths"):
            label_path = self.label_paths[index % self.image_size]
            read_items["label"] = np.array([label_path])
            # read_items['label_raw'] = label_path
            # metadata['label_path'] = label_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        table_cols=[],
        image_extension="npy",
        **kwargs,
    ):
        with open(dataset_file, "rb") as f:
            dsf = pickle.load(f)
        
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _keys = [
                os.path.basename(x).split(f".{image_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))

        _c_keys = sorted(set.intersection(*all_keys))

        this_split = dsf["split"][cv_fold][self.phase]
        _filtered_keys = [x for x in _c_keys if x in this_split]

        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{image_extension}")[0] in _filtered_keys
            ]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths
        self.image_size = image_size
        print_str = f"image num: {self.image_size}"

        if len(table_cols) > 0:
            _tables = []
            for c in table_cols:
                _paths = [dsf["data"][k][c] for k in _filtered_keys]
                _tables.append(_paths)
            self.table_paths = np.array(_tables).T
            self.table_size = len(_tables[0])
            print_str += f", table num: {self.table_size}"

        if getattr(self, "label_name", None) is not None:
            _paths = [dsf["data"][k][self.label_name] for k in _filtered_keys]
            self.label_paths = np.array(_paths)
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        print(print_str)


class BoneAgeDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        table_file: str,
        label_name: str = None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.table_file = table_file
        self.label_name = label_name
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        for k, p in self.image_paths.items():
            imageX_path = p[index % self.image_size]
            read_items[k] = imageX_path
            metadata[f"{k}_path"] = imageX_path

        if hasattr(self, "table_paths"):
            table_path = self.table_paths[index % self.image_size]
            read_items["table"] = table_path

        if hasattr(self, "label_paths"):
            label_path = self.label_paths[index % self.image_size]
            read_items["label"] = label_path
            # read_items['label_raw'] = label_path
            # metadata['label_path'] = label_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        table_cols=[],
        id_name="ID",
        image_extension="npy",
        **kwargs,
    ):
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _keys = [
                os.path.basename(x).split(f".{image_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))

        dfgt = pd.read_csv(self.table_file)
        assert id_name in dfgt
        _keys = list(dfgt[id_name])
        all_keys.append(set(_keys))
        _c_keys = sorted(set.intersection(*all_keys))

        if dataset_file is None:
            if self.phase in ["train", "valid"]:
                kf = KFold(n_splits=cv_split, shuffle=True, random_state=split_seed)
                _filtered_idx = (
                    [x for x, _ in kf.split(_c_keys)][cv_fold]
                    if self.phase == "train"
                    else [x for _, x in kf.split(_c_keys)][cv_fold]
                )
                _filtered_keys = [_c_keys[i] for i in _filtered_idx]
        else:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _c_keys if x in this_split]

        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{image_extension}")[0] in _filtered_keys
            ]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths
        self.image_size = image_size
        print_str = f"image num: {self.image_size}"

        if len(table_cols) > 0:
            _tables = []
            for c in table_cols:
                _dict = {k: v for k, v in zip(dfgt[id_name], dfgt[c])}
                _paths = [_dict[k] for k in _filtered_keys]
                _tables.append(_paths)
            self.table_paths = np.array(_tables).T
            self.table_size = len(_tables[0])
            print_str += f", table num: {self.table_size}"

        if getattr(self, "label_name", None) is not None:
            _dict = {k: v for k, v in zip(dfgt[id_name], dfgt[self.label_name])}
            _paths = [_dict[k] for k in _filtered_keys]
            self.label_paths = np.array(_paths)
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        print(print_str)


class SegmentationImageDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        label_dir=None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        for k, p in self.image_paths.items():
            imageX_path = p[index % self.image_size]
            read_items[k] = imageX_path
            metadata[f"{k}_path"] = imageX_path

        if hasattr(self, "label_paths"):
            label_path = self.label_paths[index % self.image_size]
            read_items["label"] = label_path
            # read_items['label_raw'] = label_path
            metadata["label_path"] = label_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        image_extension="nii.gz",
        label_extension="nii.gz",
        **kwargs,
    ):
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _keys = [
                os.path.basename(x).split(f".{image_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))
        if getattr(self, "label_dir", None) is not None:
            _paths = sorted(
                glob.glob(os.path.join(self.label_dir, f"*.{label_extension}"))
            )
            _keys = [
                os.path.basename(x).split(f".{label_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))
        _c_keys = sorted(set.intersection(*all_keys))

        if dataset_file is None:
            if self.phase in ["train", "valid"]:
                kf = KFold(n_splits=cv_split, shuffle=True, random_state=split_seed)
                _filtered_idx = (
                    [x for x, _ in kf.split(_c_keys)][cv_fold]
                    if self.phase == "train"
                    else [x for _, x in kf.split(_c_keys)][cv_fold]
                )
                _filtered_keys = [_c_keys[i] for i in _filtered_idx]
        else:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _c_keys if x in this_split]

        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{image_extension}")[0] in _filtered_keys
            ]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths
        self.image_size = image_size
        print_str = f"image num: {self.image_size}"

        if getattr(self, "label_dir", None) is not None:
            _paths = sorted(
                glob.glob(os.path.join(self.label_dir, f"*.{label_extension}"))
            )
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{label_extension}")[0] in _filtered_keys
            ]
            self.label_paths = _paths
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        print(print_str)


class MILImageDataset_FSL(Dataset):
    """
    bag -> k images
    bag-level label
    image_dir = "bag" base dir (single dir)
    image_paths = list of list of image paths for each bag
    image_size = number of bags (e.g. len(image_paths))
    """

    def __init__(
        self,
        transform,
        image_dir: str,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_bag = self.image_paths[index % self.image_size]
        N = len(imageX_bag)
        _idx = np.arange(N).tolist()
        if self.images_per_bag > 0:
            image_pick = []
            for _ in range(self.images_per_bag // N):
                image_pick += random.sample(_idx, k=N)
            image_pick += random.sample(_idx, k=self.images_per_bag % N)
        else:
            image_pick = _idx
        imageX_paths = [imageX_bag[i] for i in image_pick]
        imageX = [self.run_transform(x) for x in imageX_paths]
        imageX = torch.stack(imageX, axis=0)
        read_items["image"] = imageX
        metadata["image_path"] = imageX_paths

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return read_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file,
        cv_fold=0,
        image_extension="png",
        images_per_bag=1,
        sample_size_label=1,
        sample_size_nolabel=1,
        **kwargs,
    ):
        with open(dataset_file, "rb") as f:
            dsf = pickle.load(f)
        # with open(dataset_file, 'r') as f:
        #    dsf = json.load(f)
        this_split = dsf["split"][cv_fold][self.phase]
        label_dict = dsf["label"][cv_fold]
        NO_LABEL = dsf["NO_LABEL"] if "NO_LABEL" in dsf else -1

        all_keys = []
        _dirs = [os.path.join(self.image_dir, x) for x in this_split]
        _keys = [this_split[i] for i, x in enumerate(_dirs) if os.path.exists(x)]
        all_keys.append(set(_keys))
        _filtered_keys = sorted(set.intersection(*all_keys))

        if self.phase == "train":
            _filtered_keys_label = [x for x in _filtered_keys if x in label_dict.keys()]
            _filtered_keys_nolabel = [
                x for x in _filtered_keys if not x in label_dict.keys()
            ]
            _filtered_keys = _filtered_keys_label + _filtered_keys_nolabel
        elif self.phase == "valid":
            _filtered_keys = [x for x in _filtered_keys if x in label_dict.keys()]

        bag_ids = _filtered_keys
        bag_dirs = [os.path.join(self.image_dir, x) for x in bag_ids]
        images_paths = [
            sorted(glob.glob(os.path.join(x, f"*.{image_extension}"))) for x in bag_dirs
        ]
        image_size = len(images_paths)

        self.image_paths = images_paths
        self.image_size = image_size
        self.images_per_bag = images_per_bag

        if self.phase == "train":
            self.labels = [
                label_dict[x] if x in _filtered_keys_label else NO_LABEL
                for x in bag_ids
            ]
            self.label_size = len(_filtered_keys_label)
            print(f"image num: {self.image_size}, label num: {self.label_size}")
        # elif self.phase == 'valid':
        else:
            count_labels = np.array([x in label_dict.keys() for x in bag_ids])
            if count_labels.all():
                self.labels = [label_dict[x] for x in bag_ids]
                print(f"image num: {self.image_size}, label num: {count_labels.sum()}")
            else:
                print(
                    f"image num: {self.image_size}, preparing without labels.. only {count_labels.sum()} labels"
                )

        # not used in FSL
        self.sample_size_label = sample_size_label
        self.sample_size_nolabel = sample_size_nolabel


class MILImageDataset_SSLv0(MILImageDataset_FSL):
    def _sampler(self, shuffle=True):
        if hasattr(self, "label_size"):
            labeled_idxs = list(range(0, self.label_size))
            unlabeled_idxs = list(range(self.label_size, self.image_size))
            return TwoStreamSampler(
                labeled_idxs,
                unlabeled_idxs,
                self.sample_size_label,
                self.sample_size_nolabel,
                shuffle,
                num_samples=self.iterations_per_epoch,
            )
        else:
            return None


class MILImageDataset_SSLv1(MILImageDataset_SSLv0):
    """
    return two images for two transforms
    """

    def __init__(
        self,
        transform,
        image_dir: str,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform2=None,
        transform_seed=None,
        transform2_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__(
            transform, image_dir, phase, iterations_per_epoch, **prepare_data_kwargs
        )
        self.prepare_transforms(transform, transform2, transform_seed, transform2_seed)

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_bag = self.image_paths[index % self.image_size]
        N = len(imageX_bag)
        _idx = np.arange(N).tolist()
        if self.images_per_bag > 0:
            image_pick = []
            for _ in range(self.images_per_bag // N):
                image_pick += random.sample(_idx, k=N)
            image_pick += random.sample(_idx, k=self.images_per_bag % N)
        else:
            image_pick = _idx
        imageX_paths = [imageX_bag[i] for i in image_pick]
        imageX = [self.run_transform(x) for x in imageX_paths]
        imageX = torch.stack(imageX, axis=0)
        read_items["image"] = imageX
        imageX2 = [self.run_transform2(x) for x in imageX_paths]
        imageX2 = torch.stack(imageX2, axis=0)
        read_items["image2"] = imageX2
        metadata["image_path"] = imageX_paths

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return read_items

    ## override this to define transforms
    def prepare_transforms(
        self, transform, transform2=None, transform_seed=None, transform2_seed=None
    ):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

        tfm2 = (
            instantiate_list(transform)
            if transform2 is None
            else instantiate_list(transform2)
        )
        self.run_transform2 = Compose(tfm2)
        if isinstance(transform2_seed, int):
            self.run_transform2.set_random_state(seed=transform2_seed)


### Single Image Dataset


class ImageDataset_FSL(Dataset):
    def __init__(
        self,
        transform,
        image_dir: str,
        phase: str = "train",
        iterations_per_epoch: int = None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform)
        self.image_dir = image_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_path = self.image_paths[index % self.image_size]
        read_items["image"] = imageX_path
        metadata["image_path"] = imageX_path

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return self.run_transform(read_items)

    ## override this to define transforms
    def prepare_transforms(self, transform):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file,
        cv_fold=0,
        image_extension="npy",
        sample_size_label=1,
        sample_size_nolabel=1,
        **kwargs,
    ):
        with open(dataset_file, "rb") as f:
            dsf = pickle.load(f)
        # with open(dataset_file, 'r') as f:
        #    dsf = json.load(f)
        this_split = dsf["split"][cv_fold][self.phase]
        label_dict = dsf["label"][cv_fold]
        NO_LABEL = dsf["NO_LABEL"] if "NO_LABEL" in dsf else -1

        all_keys = this_split
        all_paths = [
            os.path.join(self.image_dir, f"{x}.{image_extension}") for x in all_keys
        ]

        _filtered_paths = [x for x in all_paths if os.path.exists(x)]
        _filtered_keys = [
            all_keys[i] for i, x in enumerate(all_paths) if x in _filtered_paths
        ]

        if self.phase == "train":
            _filtered_keys_label = [x for x in _filtered_keys if x in label_dict.keys()]
            _filtered_keys_nolabel = [
                x for x in _filtered_keys if not x in label_dict.keys()
            ]
            _filtered_keys = _filtered_keys_label + _filtered_keys_nolabel
        elif self.phase == "valid":
            _filtered_keys = [x for x in _filtered_keys if x in label_dict.keys()]

        image_paths = [
            os.path.join(self.image_dir, f"{x}.{image_extension}")
            for x in _filtered_keys
        ]
        image_size = len(image_paths)

        self.image_paths = image_paths
        self.image_size = image_size

        if self.phase == "train":
            self.labels = [
                label_dict[x] if x in _filtered_keys_label else NO_LABEL
                for x in _filtered_keys
            ]
            self.label_size = len(_filtered_keys_label)
        elif self.phase == "valid":
            self.labels = [label_dict[x] for x in _filtered_keys]

        # not used in FSL
        self.sample_size_label = sample_size_label
        self.sample_size_nolabel = sample_size_nolabel


class ImageDataset_SSLv0(ImageDataset_FSL):
    def _sampler(self, shuffle=True):
        if hasattr(self, "label_size"):
            labeled_idxs = list(range(0, self.label_size))
            unlabeled_idxs = list(range(self.label_size, self.image_size))
            return TwoStreamSampler(
                labeled_idxs,
                unlabeled_idxs,
                self.sample_size_label,
                self.sample_size_nolabel,
                shuffle,
                num_samples=self.iterations_per_epoch,
            )
        else:
            return None


class ImageDataset_SSLv1(ImageDataset_SSLv0):
    """
    return two images for two transforms
    """

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_path = self.image_paths[index % self.image_size]
        read_items["image"] = imageX_path
        read_items["image2"] = imageX_path
        metadata["image_path"] = imageX_path

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return self.run_transform(read_items)
