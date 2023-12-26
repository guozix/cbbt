import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset
import pickle5 as pickle
from PIL import Image
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .oxford_pets import OxfordPets


cifar_classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


@DATASET_REGISTRY.register()
class CIFAR10_FS(DatasetBase):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
        
    def __init__(self, cfg):
        self.dataset_dir = 'cifarPytorch/cifar-10-batches-py'
        object_categories = cifar_classnames
        cls_num = len(object_categories)

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir_train = os.path.join(self.dataset_dir, "train_img")
        self.image_dir_test = os.path.join(self.dataset_dir, "test_img")
        
        os.makedirs(self.image_dir_train, exist_ok=True)
        os.makedirs(self.image_dir_test, exist_ok=True)

        ### TRAIN
        downloaded_list = self.train_list
        
        self.data = []
        self.targets = []
        self.filenames = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.dataset_dir, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
                self.filenames.extend(entry['filenames'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        shots = cfg.DATASET.NUM_SHOTS
        if shots > 0:
            # build class index to sample few-shot training samples
            print('build sampled few-shot training data')
            cls_idx_dict = {}
            for i, j in enumerate(self.targets):
                cls_idx_dict.setdefault(j, list())
                cls_idx_dict[j].append(i)
            fs_idx_for_train = []
            for i in cls_idx_dict:
                # print('fewshot cifar', i, len(cls_idx_dict[i]))
                fs_idx_for_train.extend(random.sample(cls_idx_dict[i], shots))
            
            self.data = self.data[fs_idx_for_train]
            few_target_list = []
            few_filename_list = []
            for i in fs_idx_for_train:
                few_target_list.append(self.targets[i])
                few_filename_list.append(self.filenames[i])
            self.targets = few_target_list
            self.filenames = few_filename_list

        train = []
        for i in range(len(self.filenames)):
            cur_img = Image.fromarray(self.data[i])

            cur_path = os.path.join(self.image_dir_train, self.filenames[i])
            cur_img.save(cur_path)

            # L = [0] * cls_num
            # L[self.targets[i]] = 1
            # for name in self.im_name_list_val:
            item_ = Datum(impath=cur_path, label=torch.tensor(self.targets[i]), classname='')
            train.append(item_)

        ### TEST
        downloaded_list = self.test_list
        self.data = []
        self.targets = []
        self.filenames = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.dataset_dir, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
                self.filenames.extend(entry['filenames'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC
        
        test = []
        test_labels = []
        for i in range(len(self.filenames)):
            cur_img = Image.fromarray(self.data[i])
            
            cur_path = os.path.join(self.image_dir_test, self.filenames[i])
            cur_img.save(cur_path)

            # for name in self.im_name_list_val:
            item_ = Datum(impath=cur_path, label=torch.tensor(self.targets[i]), classname='')
            test.append(item_)
        
        super().__init__(train_x=train, val=test[::100], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
        

def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x

class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)
