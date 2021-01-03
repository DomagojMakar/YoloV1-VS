import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader

from os import sys
from random import randrange

from tqdm import tqdm
from model import Yolov1
from dataset import VOCDataset
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

# seed = 58 # 123 # For testing, so we get the same dataset loading each time
torch.manual_seed(randrange(1000))

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper, gtx1070 cannot do 32
WEIGHT_DECAY = 0 # Will pretrain on imagenet (training takes up to few weeks)
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
DESIRED_ACCURACY = 0.9
TRAINING_DATASET = "data/100examples.csv"
DROP_LAST = True


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) # Improve with normalization, TDL


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
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # print(LOAD_MODEL)
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        #"data/100examples.csv",
        TRAINING_DATASET,
        transform   = transform,
        img_dir     = IMG_DIR,
        label_dir   = LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", 
        transform   = transform, 
        img_dir     = IMG_DIR, 
        label_dir   = LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        shuffle     = True,
        drop_last   = DROP_LAST,
    )

    test_loader = DataLoader(
        dataset     = test_dataset,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        shuffle     = True,
        drop_last   = True,
    )

    for epoch in range(EPOCHS):
        if LOAD_MODEL == True:
            # print("titl: " + str(train_loader))
            for x, y in train_loader:
                x = x.to(DEVICE)
                for idx in range(20):
                    bboxes = cellboxes_to_boxes(model(x))
                    bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                    print("This is bboxes: " + str(bboxes))
                    plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
              
                sys.exit()

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > DESIRED_ACCURACY:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(5)
            sys.exit(0)
            # When we get to  the desired accuracy, exit the program
            #import time
            #time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)

def ChooseTrainingSet():
    # for full dataset = train.csv, small = 8examples.csv (FOR TESTING), medium = 100examples.csv 
    global TRAINING_DATASET
    global DROP_LAST
    trainingSet = input("Choose Training set:\nInput 1 for 8TS, 2 for 100TS, or 3 for full TS: ")

    if trainingSet == "1":
        TRAINING_DATASET = "data/8examples.csv"
        DROP_LAST = False;
        # If we train with 8ex, dropping 8 will drop all of our examples
    elif trainingSet == "2":
        TRAINING_DATASET = "data/100examples.csv"
    else:
        TRAINING_DATASET = "data/train.csv"

def IsFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def DesiredAccuracy():
    global DESIRED_ACCURACY
    while True:
        desiredAccuracy = input("Enter the desired accuracy of the training model (rec: 0.85-0.99): ")

        if IsFloat(desiredAccuracy) and float(desiredAccuracy) > 0 and float(desiredAccuracy) < 1:
            DESIRED_ACCURACY = float(desiredAccuracy)
            break

    ChooseTrainingSet()


def TrainModel():
    global LOAD_MODEL

    while True:
        trainModel = input("Do you want to train the model? Y/N: ").upper()
        # print(trainModel)  
        if trainModel == "Y":
            LOAD_MODEL = False
            # print(LOAD_MODEL)
            DesiredAccuracy()
            break
        elif trainModel == "N":
            LOAD_MODEL = True
            break


if __name__ == "__main__":
    TrainModel()

    main()



