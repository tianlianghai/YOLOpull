"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from torchvision import models
from torch import nn
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import logging 

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
PHASE = 'train'
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 5e-4
EPOCHS = 150
MOMENTUM=0.9
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL_FILE = "voc.pth"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
def get_vgg():
    net = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
    net.classifier[-1] = nn.Linear(4096, 7*7*30)

    net.to('cuda')
    return net

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    logging.info(f"Mean loss {sum(mean_loss) / len(mean_loss)}")


def main():
    #model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    model = get_vgg()
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
    )
    loss_fn = YoloLoss()
    logging.basicConfig(filename="yolo.log", encoding='utf-8', level=logging.DEBUG, format='[%(levelname)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    if PHASE !='train':
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    
    best_map = 0
    best_model =  {
                   "state_dict": model.state_dict(),
                   "optimizer": optimizer.state_dict(),
               }
    for epoch in range(EPOCHS):
        # this part is for showing picture in the ipynb
        if PHASE=='show':
            for x, y in test_loader:
                x = x.to(DEVICE)
                bboxes = cellboxes_to_boxes(model(x))
                for idx in range(8):
                    
                    bboxes_nms = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                    plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes_nms)
                
                exit()
        
        if PHASE=='test':
            logging.info(f"############on TEST###################")

            logging.info(f"getting boxes")
            pred_boxes, target_boxes = get_bboxes(
                test_loader, model, iou_threshold=0.5, threshold=0.4
            )

            logging.info("calculating AP")
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Test mAP: {mean_avg_prec}")
            logging.info(f"Test mAP: {mean_avg_prec}")
        
            logging.info(f"exiting")
            print("exiting")
            return
        # else if on train phase
        logging.info(f"########################EPOCH {epoch}/{EPOCHS} #####################") 
        train_fn(train_loader, model, optimizer, loss_fn)
        
        if epoch > 20 and epoch % 5 == 0:
            logging.info("############# on TRAIN##############")
            logging.info("getting boxes")
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4
            )

            logging.info("calculating AP")
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )

            if mean_avg_prec > 0.9:
               checkpoint = {
                   "state_dict": model.state_dict(),
                   "optimizer": optimizer.state_dict(),
               }
               save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
               return
            print(f"Train mAP: {mean_avg_prec}")
            logging.info(f"Train mAP: {mean_avg_prec}")
            
            logging.info(f"############on TEST###################")

            logging.info(f"getting boxes")
            pred_boxes, target_boxes = get_bboxes(
                test_loader, model, iou_threshold=0.5, threshold=0.4
            )

            logging.info("calculating AP")
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Test mAP: {mean_avg_prec}")
            if mean_avg_prec > best_map:
                best_map = mean_avg_prec
                best_model= {
                   "state_dict": model.state_dict(),
                   "optimizer": optimizer.state_dict(),
               }
            logging.info(f"Test mAP: {mean_avg_prec}")
    
    logging.info(f"best mAP on testset is {best_map}")
    save_checkpoint(best_model, filename=LOAD_MODEL_FILE)



if __name__ == "__main__":
    try:
        main()
    except Exception() as e:
        logging.error(e)
