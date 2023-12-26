import os
import pickle
from tqdm import tqdm
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        print("||||self.dataset_dir", self.dataset_dir)
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            # get {cls_idx: folder}
            idx2foldername = self.read_idx2folder(os.path.join(self.image_dir, "meta/train.txt"))
            train = self.read_data(classnames, idx2foldername, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data_val(classnames, idx2foldername, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, idx2clsdir, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        items = []

        for label in tqdm(range(1000), desc="read_data train"):
            folder = idx2clsdir[label]

            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
    
    def read_data_val(self, classnames, idx2clsdir, split_dir):
        split_img_dir = os.path.join(self.image_dir, split_dir)
        items = []

        with open(os.path.join(self.image_dir, f"meta/{split_dir}.txt"), "r") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="read_data_val"):
                line = line.strip().split(" ")
                imname = line[0]
                label = int(line[1])
                classname = classnames[idx2clsdir[label]]
                impath = os.path.join(split_img_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
        return items

    def read_idx2folder(self, train_label_file):
        idx2folder = {}
        with open(train_label_file, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="read_idx2folder"):
                line = line.strip().split(" ")
                folder = line[0].split('/')[0]
                label_idx = int(line[1])
                if label_idx in idx2folder:
                    assert folder == idx2folder[label_idx]
                else:
                    idx2folder[label_idx] = folder
        
        for i in range(1000):
            assert i in idx2folder
        return idx2folder
