data_dir = "/path/to/datasets"
fraction = 1.0
save_dir = "./runs/out"
lr = 0.02
init_lr = 1 * lr/8 * 1    # 0.02/8 * num_gpu (if batch_size is 2)
num_epochs = 26
batch_size_train = 1 * 2
batch_size_val = 1
momentum = 0.9
weight_decay = 1e-4
lr_milestones = [16, 22]
lr_gamma = 0.1
do_warmup = True
backbone = "fasterrcnn_resnet50_fpn"
backbone_weights = f"./external/vision/references/detection/checkpoints/{backbone}/checkpoint.pth"
mean = [0.485, 0.456, 0.406]    # None
std = [0.229, 0.224, 0.225]     # None
gpu_ids = [0]
clear_tensorboard = False
resume = None  # "./runs/fasterrcnn_resnet50_fpn/checkpoint.pth"
test_only = False
test_baseline = False
inference = None  # "/path/to/images"

# Names
# fasterrcnn_mobilenet_v3_large_320_fpn
# fasterrcnn_mobilenet_v3_large_fpn
# fasterrcnn_resnet50_fpn
# fasterrcnn_resnet50_fpn_v2
# fcos_resnet50_fpn
# retinanet_resnet50_fpn
# retinanet_resnet50_fpn_v2
# ssd300_vgg16
