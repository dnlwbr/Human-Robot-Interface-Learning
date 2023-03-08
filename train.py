import copy
from contextlib import redirect_stdout
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
from datetime import timedelta
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import utils


def train_model(loaders_dict, model, criterion, optimizer, scheduler, do_warmup,
                num_epochs, save_dir, tb_writer, device, resume):

    # Add model graph to Tensorboard
    utils.add_model_to_tensorboard(tb_writer, loaders_dict["train"], model, device)

    best_mean_ap = 0.0
    start_epoch, best_epoch = 0, 0
    if resume:
        model, optimizer, lr_scheduler, start_epoch, best_mean_ap = utils.load_checkpoint(resume, model, optimizer, scheduler, device)
    best_model_wts = copy.deepcopy(model.state_dict())

    start = time.time()
    for epoch in range(start_epoch, num_epochs):
        train_epoch(loaders_dict, epoch, num_epochs, model, device,
                    criterion, optimizer, scheduler, do_warmup, tb_writer)
        mean_ap = val_epoch(loaders_dict, epoch, num_epochs, model, device, tb_writer)
        if mean_ap > best_mean_ap:
            # deep copy the model
            best_mean_ap = mean_ap
            best_epoch = epoch
            utils.save_checkpoint(save_dir, model, optimizer, scheduler, epoch, best_mean_ap)
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start
    print(f"Training completed in {timedelta(seconds=time_elapsed)}")
    print(f"Best AP@[IoU=0.50:0.95 | area=all | maxDets=100]: {best_mean_ap:.4f} in epoch {best_epoch + 1}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_epoch(loaders_dict, epoch, num_epochs, model, device, criterion, optimizer, scheduler, do_warmup, tb_writer):
    model.train()  # Set model to training mode
    data_loader = loaders_dict['train']

    warmup_scheduler = None
    if do_warmup and epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    # Iterate over data.
    running_loss = 0.0
    with tqdm(data_loader, unit=" batch") as tepoch:
        for i, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Epoch [{epoch + 1}/{num_epochs}] (train)")

            inputs = utils.move_to_device(inputs, device)
            targets = utils.move_to_device(targets, device)

            # Forward pass
            loss_dict = model(inputs, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * len(inputs)

            if i+1 < len(tepoch):
                tepoch.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']}")
            else:
                epoch_loss = running_loss / len(data_loader.dataset)
                tepoch.set_postfix(epoch_loss=f"{epoch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']}")
                with tb_writer:
                    tb_writer.add_scalar(f'Loss (train)', epoch_loss, epoch)

            if warmup_scheduler is not None:
                warmup_scheduler.step()

        if scheduler is not None:
            scheduler.step()


def val_epoch(loaders_dict, epoch, num_epochs, model, device, tb_writer):
    model.eval()  # Set model to evaluate mode
    data_loader = loaders_dict['val']

    # Iterate over data.
    img_ids, results = [], []
    with tqdm(data_loader, unit=" batch") as tepoch:
        for i, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Epoch [{epoch + 1}/{num_epochs}] (val)")

            inputs = utils.move_to_device(inputs, device)

            # Forward pass
            with torch.no_grad():  # track history only if in train
                predictions = model(inputs)

            # Statistics
            res = {target["image_id"]: pred for target, pred in zip(targets, predictions)}
            img_ids_batch = list(np.unique(list(res.keys())))
            img_ids.extend(img_ids_batch)
            catNms = loaders_dict["train"].dataset.get_classes()
            results.extend(utils.prepare_for_coco_detection(res, data_loader, catNms))

            if i + 1 == len(tepoch):
                with redirect_stdout(io.StringIO()) as f:  # Suppress console output
                    coco_gt = data_loader.dataset.coco
                    coco_dt = COCO.loadRes(coco_gt, results) if results else COCO()
                    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
                    coco_eval.params.imgIds = img_ids   # coco_gt.getImgIds()
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                utils.log_coco_eval_stats(tb_writer, coco_eval, epoch)
                mean_ap = coco_eval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
                tepoch.set_postfix(mAP=f"{mean_ap:.4f}")

    # Print details if desired
    # print(f.getvalue())

    return mean_ap
