from contextlib import redirect_stdout
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
from tqdm import trange, tqdm

import torch

import utils


def test_model(loaders_dict, model, tb_writer, device, catNms=None):
    start = time.time()
    model.eval()  # Set model to evaluate mode
    data_loader = loaders_dict["test"]

    # Iterate over data.
    img_ids, results = [], []
    with tqdm(data_loader, unit=" batch") as tepoch:
        for i, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Test")

            inputs = utils.move_to_device(inputs, device)

            # Forward pass
            with torch.no_grad():  # track history only if in train
                predictions = model(inputs)

            # Statistics
            res = {target["image_id"]: pred for target, pred in zip(targets, predictions)}
            img_ids_batch = list(np.unique(list(res.keys())))
            img_ids.extend(img_ids_batch)
            results.extend(utils.prepare_for_coco_detection(res, data_loader, catNms))

    # Evaluate
    coco_gt = data_loader.dataset.coco
    coco_dt = COCO.loadRes(coco_gt, results) if results else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = img_ids  # coco_gt.getImgIds()

    # Uncomment the following line if you want to evaluate only classes that are in MS COCO too.
    # catNms = ['__background__', 'fork', 'frisbee', 'knife', 'scissors', 'table tennis ball', 'toothbrush']

    # Sort categories to be consistent in all evaluations and plots
    if catNms is not None:
        catNms = utils.sort_catNms_by_coco_catIDs(catNms, coco_gt)

    cats = [[]] if catNms is None else [*catNms, catNms]
    for i, cat in enumerate(cats):
        label = cat if i+1 < len(cats) else "Mean"
        print(f"{label}:")
        cat = cat if isinstance(cat, list) else [cat]   # https://github.com/cocodataset/cocoapi/issues/291
        coco_eval.params.catIds = coco_gt.getCatIds(catNms=cat)
        with redirect_stdout(io.StringIO()) as f:  # Suppress console output
            coco_eval.evaluate()
            coco_eval.accumulate()
        coco_eval.summarize()
        utils.write_resul_to_tensorboard(tb_writer, coco_eval, label)

    # Precision-Recall Curves
    utils.precision_recall_curve(tb_writer, coco_eval, catNms)
    utils.precision_recall_curve(tb_writer, coco_eval, catNms, iou=0.5)
    utils.precision_recall_curve(tb_writer, coco_eval, catNms, iou=0.7)
    utils.precision_recall_curve(tb_writer, coco_eval, catNms, iou=0.75)

    # Other Curves
    utils.ap_iou_curve(tb_writer, coco_eval, catNms)
    utils.recall_iou_curve(tb_writer, coco_eval, catNms)

    time_elapsed = time.time() - start
    print(f"Test completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
