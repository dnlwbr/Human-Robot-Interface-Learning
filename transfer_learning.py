import torch
import torch.optim as optim
from torchvision import transforms
import torch.utils.data

import numpy as np
import random

from data import MultiViewSet
from test import test_model
from train import train_model
from model import NeuralNet
import external.vision.references.detection.presets as presets
import utils
import mscoco_cats


def main(data_dir, fraction, init_lr, num_epochs, batch_size_train, batch_size_val,
         momentum, weight_decay, lr_milestones, lr_gamma, do_warmup, num_workers, backbone, backbone_weights,
         mean, std, save_dir, gpu_ids, clear_tensorboard, resume, test_only, test_baseline, inference):
    # Set Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Print Info
    print(f"Output folder: {save_dir}")
    print("Config:")
    for arg, value in vars(args).items():
        print(f"\t{arg} = {value}")

    # Tensorboard
    if not bool(inference):
        clear_tensorboard = False if bool(resume) else clear_tensorboard
        tb_writer = utils.tensorboard(save_dir, clear=clear_tensorboard)

    # Create dataset
    train_set = MultiViewSet(data_dir, train=True,
                             # transforms=PreprocessTransform(train=True)
                             transforms=presets.DetectionPresetTrain(data_augmentation="ssd"),
                             mean=mean, std=std, fraction=fraction)
    val_set = utils.CocoDetection(img_folder=f"{data_dir}/val/images",
                                  ann_file=f"{data_dir}/val/annotations.json",
                                  transform=transforms.ToTensor())
    test_set = utils.CocoDetection(img_folder=f"{data_dir}/test/images",
                                   ann_file=f"{data_dir}/test/annotations.json",
                                   transform=transforms.ToTensor())
    if test_baseline:
        test_set = utils.rename_coco_classes(test_set, "table tennis ball", "sports ball")
    # test_set = utils.CocoDetection(img_folder="/home/weber/data/datasets/coco/images/val2017",
    #                                ann_file="/home/weber/data/datasets/coco/annotations/instances_val2017.json",
    #                                transform=transforms.ToTensor())

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True,
                                               num_workers=num_workers, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_val, shuffle=False,
                                             num_workers=num_workers, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_val, shuffle=False,
                                              num_workers=num_workers, collate_fn=utils.collate_fn)
    loaders_dict = {"train": train_loader, "val": val_loader, "test": test_loader}   # COCO test set is not public

    # Define model
    if not test_baseline:
        backbone_model = utils.get_backbone(backbone, weights=backbone_weights, mean=train_set.mean, std=train_set.std)
        model = NeuralNet(name=backbone, backbone=backbone_model, num_classes=len(train_set.get_classes()))
    else:
        model = utils.get_backbone(backbone, mean=train_set.mean, std=train_set.std)

    # Define training strategy
    criterion = None  # nn.CrossEntropyLoss()
    # params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # Device configuration
    device, model = utils.device_configuration(gpu_ids, model)

    # Train/Validation
    if not test_only and not inference:
        model = train_model(loaders_dict, model, criterion, optimizer, lr_scheduler, do_warmup, num_epochs=num_epochs,
                            save_dir=save_dir, tb_writer=tb_writer, device=device, resume=resume)
    if (test_only or inference) and resume:
        model, _, _, _, _ = utils.load_checkpoint(resume, model, optimizer, lr_scheduler, device)

    # Test / Inference
    catNms = train_set.get_classes() if not test_baseline else mscoco_cats.catNms
    if not inference:
        test_model(loaders_dict, model, tb_writer, device, catNms=catNms)
    else:
        utils.inference(model, inference, device, model_name=backbone, catNms=catNms)


if __name__ == "__main__":
    args = utils.parse_args()
    args = utils.config_parser(args)
    main(data_dir=args.data_dir,
         fraction=args.fraction,
         save_dir=args.save_dir,
         init_lr=args.init_lr,
         num_epochs=args.num_epochs,
         batch_size_train=args.batch_size_train,
         batch_size_val=args.batch_size_val,
         momentum=args.momentum,
         weight_decay=args.weight_decay,
         lr_milestones=args.lr_milestones,
         lr_gamma=args.lr_gamma,
         do_warmup=args.do_warmup,
         num_workers=args.num_workers,
         backbone=args.backbone,
         backbone_weights=args.backbone_weights,
         mean=args.mean,
         std=args.standard_deviation,
         gpu_ids=args.gpu_ids,
         clear_tensorboard=args.clear_tensorboard,
         resume=args.resume,
         test_only=args.test_only,
         test_baseline=args.test_baseline,
         inference=args.inference)
