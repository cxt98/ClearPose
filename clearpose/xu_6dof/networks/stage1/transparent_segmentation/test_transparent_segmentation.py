import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
import argparse
import torchvision.transforms.functional as F

import clearpose.xu_6dof.networks.references.detection.transforms as T
from clearpose.xu_6dof.networks.stage1.transparent_segmentation.mask_rcnn import build_model
from clearpose.xu_6dof.datasets.transparent_segmentation_dataset import TransparentSegmentationDataset


def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(config={"num_classes": 63}):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description="Arg parser for SegmentationDataset")
    parser.add_argument(
        "-pixelthreshold", type=int, default = 200, help="minimal bbx pixel area for selecting as training sample"
    )
    parser.add_argument(
        "-root", type=str, default="./data/clearpose", help="path to root dataset directory"
    )
    parser.add_argument(
        "-model", type=str, default="./ClearPose/experiments/xu_6dof/stage1/transparent_segmentation/models/mask_rcnn_28.pt", help="path to root dataset directory"
    )

    #### For the dataset 
    args = parser.parse_args()
    config["mask_rcnn_model"] = args.model
    
    dataset_test = TransparentSegmentationDataset(dataset_name='test', transforms=get_transform(train=False), ratio=0.01)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = build_model(config)
    model.load_state_dict(torch.load(config["mask_rcnn_model"])['model_state_dict'], strict=False)
    model.eval()
    model.to(device)
    
    cpu_device = torch.device("cpu")

    iou = []

    for images, targets in data_loader_test:
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        masks = outputs[0]['masks'][outputs[0]['masks'].sum(1).sum(1).sum(1)<100000]
        mask = (masks.sum(0)>0.5).cpu().numpy()[0].astype('uint8')*255

        image_path = targets[0]['image_path']
        Image.fromarray(mask).save(image_path.replace('color', 'label-predict'))
        print(image_path)
        mask = mask.astype(bool)
        gt_mask = np.array(Image.open(image_path.replace('color', 'label'))).astype(bool)
        iou.append(np.sum(gt_mask & mask) / np.sum(gt_mask | mask))
        
    print('IoU,', np.mean(iou), np.std(iou))

if __name__=="__main__":
    main()