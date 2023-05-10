import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os

## This script trains the model according to input training data, according to the parameters below.

### Parameters
batchSize = 10 			         ### Number of images to evaluate per iteration
imageSize = [1024, 1024] 	         ### Target resolution to resize input data to when training
root = os.getcwd() 	 ### Directory containing training data, must contain subdirectories "Images" and "Labels", corresponding files sharing a name
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available

### Training Data Loader, adapted code courtesy of Sagi Eppel:
### https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code

# Populate arrays containing training images/masks

imgs = list(sorted(os.listdir(os.path.join(root, "RGB_files"))))
imgs.sort()
msks = list(sorted(os.listdir(os.path.join(root, "Mask_files"))))
msks.sort()
# imgs = []
# msks = []
# for pth in os.listdir(trainDir + "/RGB_med"):
#     imgs.append(trainDir + "/RGB_med/" + pth)
#     temp = trainDir + "/Masks_med/" + pth
#     temp = temp[:-4]
#     temp += ".png" ## Semantic drone dataset labels and images are different formats for some reason
#     msks.append(temp)

def loadData():
    batch_Imgs=[]
    batch_Data=[] # load images and masks
    for i in range(batchSize):

        # load and resize image
        idx=random.randint(0,len(imgs)-1)
        img = cv2.imread(os.path.join(root, "RGB_files", imgs[idx]))
#         img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)


        masks=[]
        img_file = msks[idx]
        if os.path.isfile(os.path.join(root, "Mask_files", img_file)):
            vesMask = cv2.imread(os.path.join(root, "Mask_files", img_file), 0)
            if vesMask is not None:
                vesMask = (vesMask > 0).astype(np.uint8)
                vesMask = cv2.resize(vesMask, imageSize, cv2.INTER_NEAREST)
                masks.append(vesMask)
            else:
                print(f"Failed to read image file: {img_file}")
        else:
            print(f"Image file not found: {img_file}")

        # load masks
#         masks=[]
#         vesMask = (cv2.imread(msks[idx], 0) > 0).astype(np.uint16)  # Read instance mask
#         vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST)
#         masks.append(vesMask)# get bounding box coordinates for each mask

        num_objs = len(masks)
        if num_objs==0: return loadData() # if image have no objects just load another image
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data

### Mask R-CNN model initialization, courtesy of Sagi Eppel:
### https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

### Mask R-CNN model training loop, courtesy of Sagi Eppel:
### https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code

for i in range(10001):
            images, targets = loadData()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            print(i,'loss:', losses.item())
            if i%100==0:
                torch.save(model.state_dict(), str(i) + ".torch")
