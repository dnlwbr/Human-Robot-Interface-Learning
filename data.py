import cv2
import os
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange, tqdm
import yaml

import torch
import torchvision.utils
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

import utils
from external.vision.references.detection.presets import DetectionPresetTrain, DetectionPresetEval
from transform import PreprocessTransform
from utils import collate_fn, show_labeled


class MultiViewSet(Dataset):
    def __init__(self, path, train=True, transforms=None, mean=None, std=None,
                 is_valid_criterion="folder", fraction=1.0):

        sub_dir = "train" if train else "val"
        self.data_dir = Path(path) / sub_dir
        self.transforms = transforms
        self.is_valid_criterion = is_valid_criterion

        self.image_folder = datasets.ImageFolder(str(self.data_dir),
                                                 # transform=self.transform,
                                                 target_transform=lambda x: x+1,    # 0 is background
                                                 is_valid_file=self.is_valid_file)

        self.indices = utils.get_subset_indices(self.image_folder, fraction)
        self.num_samples = len(self.indices)

        if not train:
            # Use the training statistics, to not leak the validation and/or test dataset information into the training.
            assert (mean is not None and std is not None), "Validation set requires the specification of " \
                                                           "the mean and standard deviation of the training set!"

        if mean is None or std is None:
            print("Warning: Calculate mean and standard deviation (Old values may be overridden)")
            tmp_image_folder = datasets.ImageFolder(str(self.data_dir), torchvision.transforms.ToTensor(),
                                                    is_valid_file=self.is_valid_file)
            subset = torch.utils.data.Subset(tmp_image_folder, self.indices)
            self.mean, self.std = self.calc_mean_and_std(subset)
            print(f"mean={self.mean},\n std={self.std}")
        else:
            self.mean = np.asarray(mean)
            self.std = np.asarray(std)

    def __getitem__(self, index):
        img, label = self.image_folder[self.indices[index]]

        # Get path of roi
        img_path, _ = self.image_folder.samples[self.indices[index]]
        img_path = Path(img_path)
        parts = list(img_path.parts)
        parts[-2] = "roi"
        parts[-1] = f"{img_path.stem[:4]}_roi.yaml"
        roi_path = Path(*parts)

        # Read box from yaml
        with open(str(roi_path), "r") as file:
            try:
                yaml_file = yaml.safe_load(file)
            except yaml.YAMLError as err:
                print(err)

        x1 = yaml_file["ROI"]["x"]
        y1 = yaml_file["ROI"]["y"]
        x2 = x1 + yaml_file["ROI"]["width"]
        y2 = y1 + yaml_file["ROI"]["height"]
        box = torch.tensor([x1, y1, x2, y2])

        boxes = torch.stack((box,))  # Tensor[N,[x1, y1, x2, y2]]
        target = {"boxes": boxes, "labels": torch.tensor([label])}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # Dataset size
    def __len__(self):
        return self.num_samples

    def is_valid_file(self, path):
        assert self.is_valid_criterion in ["mode_is_rgb", "folder"], "Invalid criterion!"

        try:
            with PIL.Image.open(path) as img:
                try:
                    img.verify()
                except Exception:
                    return False
                if self.is_valid_criterion == "mode_is_rgb":
                    if img.mode == "RGB":
                        return True
        except PIL.UnidentifiedImageError:
            return False

        if self.is_valid_criterion == "folder":
            p = Path(path)
            if p.parts[-2] == "rgb":
                return True

        return False

    @staticmethod
    def calc_mean_and_std(dataset):
        num_samples = len(dataset)
        channels_sum, channels_squared_sum = 0, 0
        for i in trange(0, num_samples):
            data, _ = dataset[i]
            # Mean over (batch,) height and width, but not over the channels
            channels_sum += torch.mean(data, dim=[1, 2])
            channels_squared_sum += torch.mean(data ** 2, dim=[1, 2])

        mean = channels_sum / num_samples

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_samples - mean ** 2) ** 0.5

        return mean, std

    def get_classes(self):
        classes = ["__background__"]
        classes.extend([cls.lower() for cls in self.image_folder.classes])
        return classes

    @staticmethod
    def add_margin(box: torch.Tensor, image_size, target_interval):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        c_x = (x2 + x1) / 2
        c_y = (y2 + y1) / 2
        img_width, img_height = image_size

        # [0,1] -> [target_interval[0], target_interval[1]]
        x_factor, y_factor = 1, 1
        if width / img_width < 0.5:
            x_factor = (target_interval[1] - target_interval[0]) * (width / img_width) + target_interval[0]
        if height / img_height < 0.5:
            y_factor = (target_interval[1] - target_interval[0]) * (height / img_height) + target_interval[0]

        width *= x_factor
        height *= y_factor
        x1 = c_x - width / 2
        x2 = c_x + width / 2
        y1 = c_y - height / 2
        y2 = c_y + height / 2

        x1 = x1 if x1 > 0 else 0
        x2 = x2 if x2 < img_width else img_width
        y1 = y1 if y1 > 0 else 0
        y2 = y2 if y2 < img_height else img_height

        return torch.tensor([x1, y1, x2, y2])

    def denormalize_tensor(self, tensor):
        # Input shape can be [N, C, H, W] or [C, H, W]
        # x_normalized = (x - mean) / std
        # x_denormalized = x * std + mean
        denormalize = transforms.Normalize((-1 * self.mean / self.std), (1.0 / self.std))
        return denormalize(tensor)

    def denormalize(self, lst):
        return [self.denormalize_tensor(tensor) for tensor in lst]

    def show_batch(self, imgs, title, denormalize=False):
        """Show demo for batch of tensors."""
        imgs = self.denormalize(imgs) if denormalize else list(imgs)
        imgs = torchvision.utils.make_grid(imgs)    # Make a grid from batch (list of tensors)
        imgs = imgs.numpy().transpose((1, 2, 0))
        imgs = np.clip(imgs, 0, 1)
        plt.imshow(imgs)
        plt.title(title)
        plt.show()

    def print_stats(self):
        catNms = self.get_classes()
        d = {"label": [], "x1": [], "y1": [], "x2": [], "y2": []}
        for i in trange(0, self.num_samples):
            _, target = self.__getitem__(i)
            label = catNms[target["labels"]]
            d["label"].append(label)
            d["x1"].append(target["boxes"][0][0].item())
            d["y1"].append(target["boxes"][0][1].item())
            d["x2"].append(target["boxes"][0][2].item())
            d["y2"].append(target["boxes"][0][3].item())

        df = pd.DataFrame(data=d)
        df["width"] = df["x2"] - df["x1"]
        df["height"] = df["y2"] - df["y1"]
        df["size"] = df["width"] * df["height"]

        pd.set_option('display.max_columns', 500)
        print(df.describe())

        print("Number of images per class:")
        n_cats = []
        for cat in catNms:
            print(f"\n\t{cat}: {len(df[df['label']==cat])}")
            n_cats.append(len(df[df['label'] == cat]))

        return catNms, n_cats

    @staticmethod
    def plot_class_distribution(catNms, n_cats=None, legend=False):

        if n_cats is None:
            n_cats_all = [160, 162, 121, 140, 156, 177, 144, 139, 172, 155, 164, 171, 149, 157, 169, 193, 139, 143, 150, 152]
            n_cats0 = n_cats_all[::2]
            n_cats1 = n_cats_all[1::2]
            catNms.reverse()
            n_cats0.reverse()
            n_cats1.reverse()
        else:
            n_cats0 = n_cats

        plt.barh(np.arange(len(catNms)), n_cats0, align='center')
        if n_cats is None:
            plt.barh(np.arange(len(catNms)), n_cats1, align='center', left=n_cats0)
        plt.yticks(np.arange(len(catNms)), catNms)
        plt.xlabel('Number of viewpoints')
        # plt.title('Number of Viewpoints per Category')
        plt.tick_params(
            axis='y',
            which='both',  # major und minor ticks
            left=False  # ticks auf der y-Achse (links)
        )
        if legend:
            plt.legend(labels=["Item 1", "Item 2"])
        # ax = plt.gca()
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig('num_classes.png', dpi=600, bbox_inches='tight', pad_inches=0.05, transparent=False)
        plt.show()


def visualize_depth_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)     # cv2.CV_16UC1
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    path = "/home/weber/data/datasets/object_data_v2"
    transform = {
        # "train": PreprocessTransform(train=True),
        "train": DetectionPresetTrain(data_augmentation="ssd"),
        "val": DetectionPresetEval()
    }
    # train_set = MultiViewSet(path, train=True, transforms=transform["train"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_set = MultiViewSet(path, train=True, transforms=transform["val"], mean=[0.5629, 0.5694, 0.5623], std=[0.1041, 0.1020, 0.1110])
    # val_set = MultiViewSet(path, train=False, transforms=transform["val"], mean=train_set.mean, std=train_set.std)

    print(f"Size of train set: {len(train_set)}")
    # print(f"Size of val set: {len(val_set)}")
    print(f"Classes: {train_set.get_classes()}")

    # Show a batch of training data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    inputs, targets = next(iter(train_loader))
    # train_set.show_batch(inputs, title=[train_set.get_classes()[t["labels"]] for t in targets])
    # show_labeled_batch(inputs, targets, train_set.get_classes(), draw_box=True)
    # show_labeled(inputs[0], targets[0], train_set.get_classes())
    # show_labeled(inputs[1], targets[1], train_set.get_classes())
    # show_labeled(inputs[2], targets[2], train_set.get_classes())
    # show_labeled(inputs[3], targets[3], train_set.get_classes())

    # catNms, n_cats = train_set.print_stats()
    catNms = train_set.get_classes()[1:]
    # n_cats = [322, 261, 333, 283, 327, 335, 306, 362, 282, 302]
    train_set.plot_class_distribution(catNms)#, n_cats)

    # visualize_depth_image("/home/weber/data/datasets/object_data/train/gamepad/gamepad05/depth/0172_depth.png")
