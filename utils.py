import glob
import os
import argparse
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import models
import torchvision.transforms.functional as TF

from torchvision.io.image import read_image, write_jpeg
from torchvision.utils import draw_bounding_boxes

import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir',
                        default=config.data_dir,
                        type=str,
                        help='Path to the folder with data/datasets')
    parser.add_argument('-f', '--fraction',
                        default=config.fraction,
                        type=float,
                        help='Fraction of training set to use')
    parser.add_argument('-sd', '--save_dir',
                        default=config.save_dir,
                        type=str,
                        help='Path to the folder where output data will be saved')
    parser.add_argument('-lr', '--init_lr',
                        default=config.init_lr,
                        type=float,
                        help='Initial learning rate')
    parser.add_argument('-n', '--num_epochs',
                        default=config.num_epochs,
                        type=int,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch_size_train',
                        default=config.batch_size_train,
                        type=int,
                        help='Batch size during training')
    parser.add_argument('-bv', '--batch_size_val',
                        default=config.batch_size_val,
                        type=int,
                        help='Batch size during validation')
    parser.add_argument('-mom', '--momentum',
                        default=config.momentum,
                        type=int,
                        help='Momentum of the optimizer')
    parser.add_argument('-wd', '--weight_decay',
                        default=config.weight_decay,
                        type=int,
                        help='Weight decay of the optimizer')
    parser.add_argument('--lr_milestones',
                        default=config.lr_milestones,
                        nargs='+',
                        type=int,
                        help='List of epoch indices at which to decay learning of MultiStepLR scheduler')
    parser.add_argument('--lr_gamma',
                        default=config.lr_gamma,
                        type=float,
                        help='Multiplicative factor of learning rate decay for MultiStepLR scheduler')
    parser.add_argument('--do_warmup',
                        default=config.do_warmup,
                        type=bool,
                        help='If true, use LinearLR scheduler during first epoch')
    parser.add_argument('-nw', '--num_workers',
                        default=int(os.cpu_count() / 2) - 1,
                        type=int,
                        help='Number of workers for dataloading')
    parser.add_argument('-bb', '--backbone',
                        type=str,
                        default=config.backbone,
                        help='Used backbone model.')
    parser.add_argument('-bbw', '--backbone_weights',
                        type=str,
                        default=config.backbone_weights,
                        help='Path to weights of backbone model. If None, default pytorch weights are used.')
    parser.add_argument('-m', '--mean',
                        type=float,
                        nargs="+",
                        default=config.mean,
                        help='Mean of training data.')
    parser.add_argument('-std', '--standard_deviation',
                        type=float,
                        nargs="+",
                        default=config.std,
                        help='Standard deviation of training data.')
    parser.add_argument('-c', '--config',
                        default=None,
                        type=str,
                        help='Path to the config file to be loaded')
    parser.add_argument('-g', '--gpu_ids',
                        default=config.gpu_ids,
                        type=int,
                        nargs="*",
                        help='Used GPU device(s) (Empty=all, [-1,...]=cpu)')
    parser.add_argument('-ct', '--clear_tensorboard',
                        default=config.clear_tensorboard,
                        type=bool,
                        help='Clear/Keep tensorboard folder (never cleared when resuming)')
    parser.add_argument("--resume",
                        default=config.resume,
                        type=str,
                        help="Path of checkpoint file")
    parser.add_argument("--test_only",
                        default=config.test_only,
                        type=bool,
                        help="Perform testing only, no train/val")
    parser.add_argument("--test_baseline",
                        default=config.test_baseline,
                        type=bool,
                        help="Test torchvision baseline model pretrained on MS COCO without any changes")
    parser.add_argument("--inference",
                        default=config.inference,
                        type=str,
                        help="Path to image use for inference")
    # TODO aspect_ratio_group_factor?
    args = parser.parse_args()
    if args.test_baseline:
        args.test_only = True
        assert args.resume is None, "Baseline does not require weights!"
    if args.inference:
        if not args.test_baseline:
            assert args.resume is not None, "Inference requires path to checkpoints!"
    return args


def get_backbone(backbone, weights=None, mean=None, std=None):
    """"
    See https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
    """
    if backbone == "fasterrcnn_resnet50_fpn":
        default_weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        backbone = models.detection.fasterrcnn_resnet50_fpn(weights=default_weights, image_mean=mean, image_std=std)
        print("Backbone: Faster R-CNN (ResNet-50-FPN backbone)")
    elif backbone == "fasterrcnn_resnet50_fpn_v2":
        default_weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        backbone = models.detection.fasterrcnn_resnet50_fpn_v2(weights=default_weights, image_mean=mean, image_std=std)
        print("Backbone: Faster R-CNN v2 (ResNet-50-FPN backbone)")
    elif backbone == "fasterrcnn_mobilenet_v3_large_fpn":
        default_weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        backbone = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=default_weights, image_mean=mean,
                                                                      image_std=std)
        print("Backbone: High resolution Faster R-CNN (MobileNetV3-Large FPN backbone)")
    elif backbone == "fasterrcnn_mobilenet_v3_large_320_fpn":
        default_weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        backbone = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=default_weights, image_mean=mean,
                                                                          image_std=std)
        print("Backbone: Low resolution Faster R-CNN (MobileNetV3-Large FPN backbone) tuned for mobile use-cases")
    elif backbone == "fcos_resnet50_fpn":
        default_weights = models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT
        backbone = models.detection.fcos_resnet50_fpn(weights=default_weights, image_mean=mean, image_std=std)
        print("Backbone: FCOS (ResNet-50-FPN backbone)")
    elif backbone == "retinanet_resnet50_fpn":
        default_weights = models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        backbone = models.detection.retinanet_resnet50_fpn(weights=default_weights, image_mean=mean, image_std=std)
        print("Backbone: RetinaNet (ResNet-50-FPN backbone)")
    elif backbone == "retinanet_resnet50_fpn_v2":
        default_weights = models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        backbone = models.detection.retinanet_resnet50_fpn_v2(weights=default_weights, image_mean=mean, image_std=std)
        print("Backbone: RetinaNet v2 (ResNet-50-FPN backbone)")
    elif backbone == "ssd300_vgg16":
        default_weights = models.detection.SSD300_VGG16_Weights.DEFAULT
        backbone = models.detection.ssd300_vgg16(weights=default_weights, image_mean=mean, image_std=std)
        print("Backbone: SSD300 (VGG16 backbone)")
    elif backbone == "ssdlite320_mobilenet_v3_large":
        default_weights = models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        backbone = models.detection.ssdlite320_mobilenet_v3_large(weights=default_weights, trainable_backbone_layers=None,
                                                                  image_mean=mean, image_std=std)
        print("Backbone: SSDlite330 (MobileNetV3-Large backbone)")
    else:
        raise ValueError("Invalid backbone!")
    if weights:
        print("Loads custom backbone weights")
        backbone = load_backbone_weights(weights, backbone)
    return backbone


def load_backbone_weights(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # args = checkpoint['args']
    # epoch = checkpoint['epoch']
    return model


def get_device(device_ids):
    use_cuda = not (len(device_ids) > 0 and device_ids[0] == -1)
    if torch.cuda.is_available() and use_cuda:
        if len(device_ids) == 1:
            device = torch.device(f"cuda:{device_ids[0]}")
        else:
            device = torch.device(f"cuda")
    else:
        device = torch.device("cpu")

    return device


def device_configuration(gpu_ids, model):
    device = get_device(gpu_ids)
    print(f"Use device: {device}")
    model.to(device)
    if torch.cuda.device_count() > 1 and len(gpu_ids) <= torch.cuda.device_count() \
            and len(gpu_ids) != 1 and not (len(gpu_ids) > 0 and gpu_ids[0] == -1):
        num_gpus = torch.cuda.device_count() if len(gpu_ids) == 0 else len(gpu_ids)
        print(f"Use {num_gpus} GPUs (ids: {gpu_ids}).")
        device_ids = None if len(gpu_ids) == 0 else gpu_ids
        model = nn.DataParallel(model, device_ids=device_ids)
    return device, model


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for key, value in obj.items():
            res[key] = move_to_device(value, device)
        return res
    elif isinstance(obj, (list, tuple)):
        res = []
        for elem in obj:
            res.append(move_to_device(elem, device))
        return res
    else:
        raise TypeError(f"Invalid type {type(obj)}")


def save_checkpoint(path, model, optimizer, lr_scheduler, epoch, mean_ap):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model_state": model_state,
        "optim_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "epoch": epoch,
        "mean_ap": mean_ap
    }
    torch.save(checkpoint, p / f"checkpoint.pth")


def load_checkpoint(path, model, optimizer, lr_scheduler, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    mean_ap = checkpoint["mean_ap"]
    # Start epoch = printed epoch = epoch + 1, since first epoch starts at 0
    print(f"Checkpoint loaded (epoch={checkpoint['epoch'] + 1}, mAP={mean_ap:.4f})")
    return model, optimizer, lr_scheduler, start_epoch, mean_ap


def config_parser(args):
    if args.config is None:
        if args.inference is None:
            backup_cfg(args.save_dir, args)
    else:
        args = load_cfg(args.config, args)
        backup_cfg(args.save_dir, args)
    return args


def backup_cfg(path, args):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    cfg_path = os.path.join(path, 'config.json')
    with open(cfg_path, 'w') as file:
        json.dump(args.__dict__, file, indent=2)


def load_cfg(cfg_path, args):
    with open(cfg_path, 'r') as file:
        args.__dict__ = json.load(file)
    print(f"Config file loaded from {cfg_path}")
    return args


def tensorboard(path, clear=False):
    p = Path(path) / "tensorboard"
    if clear and p.is_dir():
        print("Clear tensorboard folder")
        shutil.rmtree(str(p))
    tb_writer = SummaryWriter(str(p))
    return tb_writer


def add_model_to_tensorboard(tb_writer, data_loader, model, device):
    """Adds graph of model to tensorboard"""
    example_img, example_target = next(iter(data_loader))
    # example_img = torch.stack(example_img)  # List[Tensor[C, H, W]] -> Tensor[N, C, H, W]
    example_img = move_to_device(example_img, device)
    example_target = move_to_device(example_target, device)
    if isinstance(model, nn.DataParallel):
        with tb_writer:
            tb_writer.add_graph(model.module, example_img)
    else:
        with tb_writer:
            pass
            # TODO fix
            # tb_writer.add_graph(model, (example_img, example_target))


def collate_fn(batch):
    return list(zip(*batch))


def get_subset_indices(data_set, fraction):
    # fraction controls the proportion of use data, e.g. fraction=0.75 -> len(subset) = 0.75 * len(data_set)
    new_length = int(len(data_set) * fraction)
    idx = list(np.linspace(0, len(data_set), new_length, endpoint=False, dtype=int))
    # subset = torch.utils.data.Subset(data_set, idx)
    return idx


def show_labeled_batch(imgs, targets, class_names, score_threshold=0, draw_box=True):
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if not isinstance(targets, (list, tuple)):
        targets = [targets]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, (img, target) in enumerate(zip(imgs, targets)):
        # img = img.detach()
        img = TF.convert_image_dtype(img, dtype=torch.uint8)
        if "scores" in target and score_threshold > 0:
            boxes = target["boxes"][target["scores"] > 0]
        else:
            boxes = target["boxes"]
        labels = [class_names[target["labels"]]]
        if draw_box:
            img = torchvision.utils.draw_bounding_boxes(img, boxes, labels=labels, width=5,
                                                        font="Ubuntu-R.ttf", font_size=30)
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_labeled(img, target, class_names):
    colormap = {"frisbee": (31, 120, 180),
                "fork": (51, 160, 44),
                "gamepad": (227, 26, 28),
                "hole puncher": (255, 127, 0),
                "knife": (106, 61, 154),
                "scissors": (166, 206, 227),
                "shuttlecock": (178, 223, 138),
                "sports ball": (251, 154, 153),
                "stapler": (253, 191, 111),
                "toothbrush": (202, 178, 214),
                "table tennis ball": (251, 154, 153),
                "others": (0, 0, 0)}

    img = TF.convert_image_dtype(img, dtype=torch.uint8)
    box = target["boxes"]
    label = class_names[target["labels"]]
    color = colormap[label] if label in colormap.keys() else colormap["others"]

    img_out = draw_bounding_boxes(img, boxes=box, labels=[label],
                                  colors=color, width=5, font="Ubuntu-R.ttf", font_size=40)
    im = TF.to_pil_image(img_out)
    im.show()


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    We need this class, because torchvision.datasets.CocoDetection has no way of obtaining the image_id from the
    raw annotations, which are needed for evaluation.
    See Issue https://github.com/pytorch/vision/issues/2720#issuecomment-700047393
    """

    def __init__(self, img_folder, ann_file, transform):
        super().__init__(img_folder, ann_file)
        self._transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transform is not None:
            img = self._transform(img)
        return img, target


def rename_coco_classes(dataset, current_name: str, output_name: str):
    k = None
    for key, value in dataset.coco.cats.items():
        if value["name"] == current_name:
            k = key
    dataset.coco.cats[k]["name"] = output_name
    for cat in dataset.coco.dataset["categories"]:
        if cat["name"] == current_name:
            cat["name"] = output_name
    return dataset


def prepare_for_coco_detection(predictions, data_loader, trainNms):
    """See https://github.com/pytorch/vision/blob/fc63f82890651bb19f27e33eefb52465aeb1c25d/references/detection/coco_eval.py#L67"""
    coco_gt = data_loader.dataset.coco
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": coco_gt.getCatIds(catNms=[trainNms[labels[k]]])[0],  # map custom ids to coco cat ids
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes) if len(coco_gt.getCatIds(catNms=[trainNms[labels[k]]])) > 0
            ]
        )
    return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def log_coco_eval_stats(tb_writer, coco_eval, epoch):
    with tb_writer:
        header = "Average Precision (val)"
        tb_writer.add_scalar(f'{header}/AP@0.50:.05:0.95', coco_eval.stats[0].item(), epoch)
        tb_writer.add_scalar(f'{header}/AP@0.5', coco_eval.stats[1].item(), epoch)
        tb_writer.add_scalar(f'{header}/AP@0.75', coco_eval.stats[2].item(), epoch)
        header = "Average Precision Across Scales (val)"
        tb_writer.add_scalar(f'{header}/AP@small', coco_eval.stats[3].item(), epoch)
        tb_writer.add_scalar(f'{header}/AP@medium', coco_eval.stats[4].item(), epoch)
        tb_writer.add_scalar(f'{header}/AP@large', coco_eval.stats[5].item(), epoch)
        header = "Average Recall (val)"
        tb_writer.add_scalar(f'{header}/AR@max=1', coco_eval.stats[6].item(), epoch)
        tb_writer.add_scalar(f'{header}/AR@max=10', coco_eval.stats[7].item(), epoch)
        tb_writer.add_scalar(f'{header}/AR@max=100', coco_eval.stats[8].item(), epoch)
        header = "Average Recall Across Scales (val)"
        tb_writer.add_scalar(f'{header}/AR@small', coco_eval.stats[9].item(), epoch)
        tb_writer.add_scalar(f'{header}/AR@medium', coco_eval.stats[10].item(), epoch)
        tb_writer.add_scalar(f'{header}/AR@large', coco_eval.stats[11].item(), epoch)


def write_resul_to_tensorboard(tb_writer, coco_eval, label=None):
    with tb_writer:
        tb_writer.add_hparams(
            {'AP@0.50:.05:0.95': coco_eval.stats[0].item(),
             'AP@0.5': coco_eval.stats[1].item(),
             'AP@0.75': coco_eval.stats[2].item(),
             'AP@small': coco_eval.stats[3].item(),
             'AP@medium': coco_eval.stats[4].item(),
             'AP@large': coco_eval.stats[5].item(),
             'AR@max=1': coco_eval.stats[6].item(),
             'AR@max=10': coco_eval.stats[7].item(),
             'AR@max=100': coco_eval.stats[8].item(),
             'AR@small': coco_eval.stats[9].item(),
             'AR@medium': coco_eval.stats[10].item(),
             'AR@large': coco_eval.stats[11].item()},
            dict(),
            run_name=label)


def precision_recall_curve(tb_writer, coco_eval, catNms, iou=-1):
    recall_thrs = coco_eval.params.recThrs      # [0:.01:1] R=101
    precision = coco_eval.eval['precision']     # [TxRxKxAxM], T=10, A=4, M=3
    precision = precision[:, :, :, 0, 2]        # area = all [all, small, medium, large], maxDets = 100 [1, 10, 100]

    iou = round((iou - 0.5) / 0.05)
    if iou < 0 or iou > 9:
        precision = np.mean(precision, axis=0)  # P@[0.5:0.05:0.95]
        title = "Precision-Recall Curve [IoU=0.50:0.95]"    # | area=all | maxDets=100]"
        iou_postfix = ""
    else:
        precision = precision[iou, :, :]  # P@[iou]
        title = f"Precision-Recall Curve [IoU={0.5 + iou * 0.05}]"
        iou_postfix = f"_IoU={0.5 + iou * 0.05}"

    mean_precision = np.mean(precision, axis=1)

    path_out = Path(tb_writer.log_dir).parent
    path_plots = path_out / "plots"
    path_plots.mkdir(parents=False, exist_ok=True)
    #  T - IoU thresholds, T=10, [.5:.05:.95]
    #  R - recall thresholds, R=101, [0:.01:1]
    #  K - cat ids
    #  A - object area ranges, A=4, [all, small, medium, large]
    #  M - thresholds on max detections per image, M=3, [1 10 100]
    with open(f"{path_out}/precisions.npy", "wb") as file:
        # [TxRxKxAxM]
        np.save(file, coco_eval.eval['precision'])
    with open(f"{path_out}/recalls.npy", "wb") as file:
        # [TxKxAxM]
        np.save(file, coco_eval.eval['recall'])

    fig = plt.figure(layout="tight")
    # plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    for i, cat in enumerate(catNms):
        plt.plot(recall_thrs, precision[:, i], label=cat)
    plt.plot(recall_thrs, mean_precision, label="mean", color="navy")

    plt.legend()
    plt.savefig(f"{path_plots}/pr_curve{iou_postfix}.jpg", dpi=600)
    with tb_writer:
        tb_writer.add_figure(title, fig, 0)


def ap_iou_curve(tb_writer, coco_eval, catNms):
    ious_thrs = coco_eval.params.iouThrs        # [.5:.05:.95] T=10
    precision = coco_eval.eval['precision']     # [TxRxKxAxM], T=10, A=4, M=3
    precision = precision[:, :, :, 0, 2]        # area = all [all, small, medium, large], maxDets = 100 [1, 10, 100]
    if len(precision[precision == -1]) > 0:
        print("Warning: Invalid values affect results.")
    average_precision = np.mean(precision, axis=1)

    path_out = Path(tb_writer.log_dir).parent
    path_plots = path_out / "plots"
    path_plots.mkdir(parents=False, exist_ok=True)

    fig = plt.figure(layout="tight")
    title = "AP-IoU-Curve"
    # plt.title("")
    plt.xlabel("IoU")
    plt.ylabel("AP")
    markers = itertools.cycle(('v', '^', '<', '>', 's', 'D', 'p', 'o', '*', 'X', '.'))
    for i, cat in enumerate(catNms):
        plt.plot(ious_thrs, average_precision[:, i], label=cat, marker=next(markers))
    plt.plot(ious_thrs, np.mean(average_precision, axis=1), label="mean", color="navy", marker=next(markers))

    plt.legend()
    plt.savefig(f"{path_plots}/ap_iou_curve.jpg", dpi=600)
    with tb_writer:
        tb_writer.add_figure(title, fig, 0)


def recall_iou_curve(tb_writer, coco_eval, catNms):
    ious_thrs = coco_eval.params.iouThrs    # [.5:.05:.95] T=10
    recall = coco_eval.eval['recall']    # [TxKxAxM], T=10, A=4, M=3
    recall = recall[:, :, 0, 2]       # area = all [all, small, medium, large], maxDets = 100 [1, 10, 100]

    path_out = Path(tb_writer.log_dir).parent
    path_plots = path_out / "plots"
    path_plots.mkdir(parents=False, exist_ok=True)

    fig = plt.figure(layout="tight")
    title = "Recall-IoU-Curve"
    # plt.title("")
    plt.xlabel("IoU")
    plt.ylabel("Recall")
    markers = itertools.cycle(('v', '^', '<', '>', 's', 'D', 'p', 'o', '*', 'X', '.'))
    for i, cat in enumerate(catNms):
        plt.plot(ious_thrs, recall[:, i], label=cat, marker=next(markers))
    plt.plot(ious_thrs, np.mean(recall, axis=1), label="mean", color="navy", marker=next(markers))

    plt.legend()
    plt.savefig(f"{path_plots}/recall_iou_curve.jpg", dpi=600)
    with tb_writer:
        tb_writer.add_figure(title, fig, 0)


def sort_catNms_by_coco_catIDs(catNms, coco_gt):
    cats = coco_gt.dataset["categories"]
    classes = [cat["name"] for cat in cats if cat["name"] in catNms]
    return classes


def get_weight_name(name):
    weights = ["FasterRCNN_ResNet50_FPN",
               "FasterRCNN_ResNet50_FPN_V2",
               "FasterRCNN_MobileNet_V3_Large_FPN",
               "FasterRCNN_MobileNet_V3_Large_320_FPN",
               "FCOS_ResNet50_FPN",
               "RetinaNet_ResNet50_FPN",
               "RetinaNet_ResNet50_FPN_V2",
               "SSD300_VGG16",
               "SSDLite320_MobileNet_V3_Large"]

    for i, weight in enumerate(weights):
        if name == weight.lower():
            return f"{weight}_Weights"

    return False


def inference(model, path, device, model_name, catNms, box_score_thresh=0.4):
    colormap = {"frisbee": (31, 120, 180),
                "fork": (51, 160, 44),
                "gamepad": (227, 26, 28),
                "hole puncher": (255, 127, 0),
                "knife": (106, 61, 154),
                "scissors": (166, 206, 227),
                "shuttlecock": (178, 223, 138),
                "sports ball": (251, 154, 153),
                "stapler": (253, 191, 111),
                "toothbrush": (202, 178, 214),
                "table tennis ball": (251, 154, 153),
                "others": (0, 0, 0)}
    model.eval()
    default_weights = getattr(models.detection, get_weight_name(model_name)).DEFAULT
    preprocess = default_weights.transforms()

    if os.path.isdir(path):
        path = Path(path)
        out = path / "out"
        out.mkdir(parents=False, exist_ok=False)
        types = ["*.jpg", "*.jpeg", "*.png"]
        img_paths = []
        for t in types:
            img_paths.extend(glob.glob(str(path / t)))
    else:
        img_paths = [path]

    for img_path in tqdm(img_paths):
        img = read_image(img_path)
        img = move_to_device(img, device)

        batch = [preprocess(img)]
        prediction = model(batch)[0]

        mask = prediction["scores"] > box_score_thresh
        # mask2 = prediction["labels"] == 9   # table tennis ball
        # mask = torch.logical_or(mask, mask2)
        for key in prediction:
            prediction[key] = prediction[key][mask]

        boxes, labels, colors = list(), list(), list()
        for i, box in zip(prediction["labels"], prediction["boxes"]):
            boxes.append(box)
            labels.append(catNms[i])
            if catNms[i] in colormap.keys():
                colors.append(colormap[catNms[i]])
            else:
                colors.append(colormap["others"])

        # boxes = prediction["boxes"]
        # labels = [catNms[i] for i in prediction["labels"]]

        box = draw_bounding_boxes(img, boxes=torch.stack(boxes),
                                  labels=labels,
                                  colors=colors,
                                  width=5,
                                  font="Ubuntu-R.ttf",
                                  font_size=40)

        if len(img_paths) == 1:
            im = TF.to_pil_image(box.detach())
            im.show()
        else:
            output_file = str(out / Path(img_path).stem) + "_labeled.jpg"
            write_jpeg(box, output_file, quality=100)
