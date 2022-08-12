from PIL import Image
import os
import random
import math
import numpy as np
import h5py
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset


class WM811K(VisionDataset):
    """
    Args:
        root (str, optional): '/home/local/eda10/jayliu/projects/wafer'.
        mode (str, optional): 'all', 'unlabel', 'train', 'test', 'train+unlabel'.
        dim  (int, optional): 32, 256.
        memory (bool, optional): If True will load entire hdf5 content into memory.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'WM-811k'
    saveType = 'all'

    def __init__(
            self,
            root: str = '/path/to/dataset',
            mode: str = 'train',
            dim:  int = 32,
            memory=True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            weighted_sampler=None,
            drs_flag=False,
            two_stage_start_epoch=None,
            indices=None,
    ) -> None:

        super(WM811K, self).__init__(root, transform=transform,
                                     target_transform=target_transform)

        self.mode = mode  # training set or test set
        self.memory = memory
        self.labeled = mode in ['train', 'test']

        self.data: Any = []
        self.targets = []

        # parameters for resampling
        self.weighted_sampler = weighted_sampler
        if self.mode != 'train':
            self.weighted_sampler = None
        self.drs_flag = drs_flag
        self.two_stage_start_epoch = two_stage_start_epoch

        # now load the picked numpy arrays
        if self.saveType == 'all':
            file_path = os.path.join(self.root, self.base_folder+'/'+str(dim), self.mode+'.h5')
            entry = h5py.File(file_path, 'r')
            if self.memory:
                self.data = entry['data'][()]
            else:
                self.data = entry['data']
            if self.labeled:
                if self.memory:
                    self.targets = entry['targets'][()]
                else:
                    self.targets = entry['targets']
            else:
                self.targets = [0] * len(self.data)
        else:
            # Deprecated. This is slow...
            file_path = os.path.join(self.root, self.base_folder+'/'+str(dim), 'all.h5')
            entry = h5py.File(file_path, 'r')
            if self.mode == 'train':
                idx = entry['train_idx']
                self.data = entry['data']
                self.data = self.data[idx]
                self.targets = entry['targets'][idx]
            elif self.mode == 'test':
                idx = entry['test_idx']
                self.data = entry['data'][idx]
                self.targets = entry['targets'][idx]
            elif self.mode == 'unlabel':
                idx = entry['unlabel_idx']
                self.data = entry['data'][idx]
                self.targets = [0] * len(self.data)
            else:
                self.data = entry['data']
                self.targets = [0] * len(self.data)

        if indices != None:
            self.data = self.data[indices]
            self.targets = self.targets[indices]

        self._load_meta()
        self.update_sampler_information()

    def _load_meta(self) -> None:
        self.classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'None']
        self.num_classes = len(self.classes)
        self.class_to_idx = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'None': 8}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if (self.weighted_sampler != None) and (not self.drs_flag or (self.drs_flag and self.epoch)):
            if self.weighted_sampler == 'balance':
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.weighted_sampler == 'square':
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            elif self.weighted_sampler == 'progressive':
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            else:
                assert False, "Invalid weighted sampler"
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        return True

    def extra_repr(self) -> str:
        return "Split: {}".format("Labeled" if self.labeled else "Unlabeled")

    def update_sampler_information(self):
        self.num_list = [0] * self.num_classes
        self.class_dict = [[] for _ in range(self.num_classes)]
        for i, target in enumerate(self.targets):
            self.num_list[target] += 1
            self.class_dict[target].append(i)

        num_list_np_array = np.array(self.num_list)
        self.instance_p = num_list_np_array / num_list_np_array.sum()

        # class balanced sampling
        max_num = max(self.num_list)
        self.class_balanced_weight = [max_num / i if i != 0 else 0 for i in self.num_list]

        # square root sampling
        sqrt_num_list = np.sqrt(num_list_np_array)
        self.square_p = sqrt_num_list / sqrt_num_list.sum()

        print(self.num_list)

    def update(self, epoch, max_epoch):
        if self.drs_flag:
            self.epoch = max(0, epoch - self.two_stage_start_epoch)
        else:
            self.epoch = epoch

        if self.weighted_sampler == "progressive":
            self.progress_p = epoch / max_epoch / self.num_classes + (1 - epoch / max_epoch) * self.instance_p
