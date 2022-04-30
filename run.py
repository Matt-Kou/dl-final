# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import argparse

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
import utils
from engine import train_one_epoch, evaluate

# from dataset import UnlabeledDataset, LabeledDataset
from dataset import LabeledDataset

from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


def faster_rcnn(backbone, num_classes=100, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        False, trainable_backbone_layers, 5, 3
    )
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model


def get_model(num_classes, backbone_path=None, pretrained_path=None, model='fasterrcnn'):
    if model == 'fasterrcnn':
        if pretrained_path:
            assert os.path.isfile(pretrained_path)
            return torch.load(pretrained_path)

        backbone = torch.load(backbone_path)
        model = faster_rcnn(backbone)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model
    raise NotImplementedError("model not implemented")

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = LabeledDataset(root=args.dataset, split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2,
                                               collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root=args.dataset, split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, shuffle=False, num_workers=2,
                                               collate_fn=utils.collate_fn)

    model = get_model(args.num_classes,
                      backbone_path=os.path.join('zoo', 'backbone', f'{args.backbone_type}-{args.backbone_subtitle}'),
                      model=args.model,
                      pretrained_path=args.pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)
        torch.save(model, os.path.join("zoo", f"{args.backbone_type}-{args.backbone_subtitle}-{args.output_subname}"
                                              f"-_ep{epoch}"))

    print("That's it!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--backbone_type', type=str, default='resnet', help='type of backbone (resnet, vit)')
    parser.add_argument('--backbone_subtitle', type=str, default='ep35', help='backbone subtitle')
    parser.add_argument('--num_classes', type=int, default=100, help='numer of classes')
    parser.add_argument('--dataset', type=str, default='/labeled', help='the path of the labeled dataset')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--output_subname', type=str, default='', help='output subname')
    parser.add_argument('--model', type=str, default='fasterrcnn', help='Faster RCNN, DETR, Detectron2')
    parser.add_argument('--pretrained', type=str, default=None, help="pretrained model")
    args = parser.parse_args()
    main(args)
