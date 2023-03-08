import os
from pathlib import Path
import PIL.Image
import random
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class PreprocessTransform:
    def __init__(self, train=True, target_height=800, target_width=1333):

        self.is_train = train
        self.random_pad = RandomPad(target_height=target_height, target_width=target_width)
        self.random_background = RandomBackground()

        # Resizing is not necessary because Faster R-CNN automatically resizes all images
        # to (800, w) or (h, 800) while maintaining the aspect ratio.  If by resizing
        # one of the dimensions exceeds 1333, this value (1333) is taken as the width/height
        # and the other dimension is then less than 800 accordingly.

        self.transform_pre_pad = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            RandomResizedRandomCrop(),
            transforms.RandomHorizontalFlip(),
            # RandomAugmentation()
        ])
        self.transform_post_pad = transforms.Compose([
            # RandomAugmentation(pre_pad=False),
            transforms.ToTensor(),
            # transforms.Normalize(self.mean, self.std)
        ])

        self.val_transform = transforms.Compose([
            # transforms.Resize(256),   # Handled by Faster-RCNN
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(self.mean, self.std) # Handled by Faster-RCNN
        ])

    def __call__(self, img):
        if self.is_train:
            img = self.transform_pre_pad(img)
            if torch.rand(1) < 0.5:
                img, box = self.random_pad(img)
            else:
                img, box = self.random_background(img)
            img = self.transform_post_pad(img)
        else:
            img = self.val_transform(img)
            box = torch.tensor([0, 0, img.shape[2], img.shape[1]])
        return img, box


class RandomResizedRandomCrop:
    def __init__(self, size_scale=(2/3, 1.0), crop_scale=(2/3, 1.0), crop_ratio=(3. / 4., 4. / 3.),
                 max_height=800, max_width=1333):
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.size_scale = size_scale
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, img):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.crop_scale, self.crop_ratio)

        # Adjust bounds of scaling factor if necessary
        size_scale = list(self.size_scale)
        max_scale = min(self.max_height / h, self.max_width / w)
        size_scale[1] = min(self.size_scale[1], max_scale)
        size_scale[0] = min(size_scale)

        # Target size maintains aspect ratio of cropped window
        random_scale = random.uniform(*size_scale)
        target_size = [int(x * random_scale) for x in [h, w]]

        assert (target_size[0] <= self.max_height and target_size[1] <= self.max_width)

        return TF.resized_crop(img, i, j, h, w, target_size)


class RandomPad:
    def __init__(self, target_height=800, target_width=1333):
        self.path_to_bg_imgs = Path("/home/weber/data/datasets/fiftyone/exported_subset/train2017")
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, img):
        p_left, p_top, p_right, p_bottom = 0, 0, 0, 0
        if img.width < self.target_width:
            p_left = random.randint(0, self.target_width - img.width)
            p_right = self.target_width - (img.width + p_left)
        if img.height < self.target_height:
            p_top = random.randint(0, self.target_height - img.height)
            p_bottom = self.target_height - (img.height + p_top)
        padding = [p_left, p_top, p_right, p_bottom]
        if torch.rand(1) < 0.5:
            mean_color = np.mean(np.asarray(img), axis=(0, 1), dtype=int)
            img = TF.pad(img, padding, tuple(mean_color), 'constant')
        else:
            img = TF.pad(img, padding, 0, 'edge')
        box = torch.tensor([padding[0], padding[1], img.width-padding[2]-1, img.height-padding[3]-1]) # [x1, y1, x2, y2]
        return img, box


class RandomBackground:
    def __init__(self):
        self.path_to_bg_imgs = Path("/home/weber/data/datasets/fiftyone/exported_subset/train2017")

    def __call__(self, img):
        bg_img = self.load_random_background()
        if bg_img.width < img.width or bg_img.height < img.height:
            box = torch.tensor([0, 0, img.width-1, img.height-1])
            return img, box
        x1 = random.randint(0, bg_img.width - img.width)
        y1 = random.randint(0, bg_img.height - img.height)
        x2 = x1 + img.width - 1
        y2 = y1 + img.height - 1
        box = torch.tensor([x1, y1, x2, y2])

        mask = self.create_mask(img)
        bg_img.paste(img, (x1, y1), mask=mask)
        img = bg_img
        return img, box

    def load_random_background(self):
        filename = random.choice(os.listdir(self.path_to_bg_imgs))
        img_path = self.path_to_bg_imgs / filename
        with open(img_path, "rb") as f:
            img = PIL.Image.open(f)
            return img.convert("RGB")

    @staticmethod
    def create_mask(img, edge_factor=0.1):
        # I_3 =s * I_1 + (1-s) * I_2
        mask = PIL.Image.new('L', img.size, 255)
        for x in range(img.width):
            for y in range(img.height):
                x_ = x if x < img.width/2 else img.width - x
                y_ = y if y < img.height/2 else img.height - y
                s = min(x_ / (img.width * edge_factor), y_ / (img.height * edge_factor))
                s = min(s, 1)
                alpha = s * 255 + (1-s) * 0
                mask.putpixel((x, y), int(alpha))
        return mask


class RandomAugmentation:
    def __init__(self, pre_pad=True):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        if pre_pad:
            self.seq = iaa.Sequential([
                sometimes(iaa.AddToHueAndSaturation((-60, 60))),  # change hue and saturation
                iaa.SomeOf((0, 4), [
                    iaa.Cutout(),  # replace one squared area within the image by a constant intensity value
                    iaa.Invert(0.2, per_channel=True),  # invert color channels
                    iaa.Add((-25, 25), per_channel=0.3),  # change brightness of images (by -10 to 10 of original value)
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                ], random_order=True)
            ], random_order=True)
        else:
            self.seq = iaa.Sequential([
                iaa.SomeOf((0, 4), [
                    #self.sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                    # convert images into their superpixel representation
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                    # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                    # sometimes move parts of the image around
                ], random_order=True),
                sometimes(iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),  # blur images using a gaussian kernel with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
                ])),
            ], random_order=True)

    def __call__(self, img):
        array = np.asarray(img)  # for png do img.convert('RGB') if necessary
        array = self.seq(image=array)
        return PIL.Image.fromarray(array)
