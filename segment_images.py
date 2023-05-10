import cv2
import numpy as np
import os
import random
import torch
import torchvision.models.segmentation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

## This script evaluates input images using the trained model, according to the parameters below.

### Parameters
imageSize = [1024, 1024]    # Target image resolution, use primarily to downsize inputs for time/memory save.
imDir = "input" 	    # Directory for input images. Directory must already exist.
outDir = "output" 	    # Directory for output segmentations. Directory must already exist.
maskColor = [128, 64, 128]  # Mask color values 0-255. 128R, 64G, 128B to be widely used for roads.
bgOn = True 		    # Toggle for including original image as background in output. Turn to False for mask only output
checkpoint = "700.torch" # Trained checkpoint file to use.


### Mask R-CNN model initialization, evaluation and output, lightly modified code courtesy of Sagi Eppel:
### https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
model.load_state_dict(torch.load(checkpoint))
model.to(device) # move model to the right device (GPU or CPU)
model.eval()

def generate_random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


for pth in os.listdir(imDir):  # Iterates through images
    images = None
    images = cv2.imread(imDir + "/" + pth)
    if images is not None:
        images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)  # Resize image

        images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)  # Preparation for evaluation
        images = images.swapaxes(1, 3).swapaxes(2, 3)
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            pred = model(images)

        im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)

        if bgOn:  # Draws background or not depending on the boolean "bgOn"
            im2 = im.copy()
        else:
            im2 = np.zeros((imageSize[1], imageSize[0], 3), dtype=im.dtype)

        for i in range(len(pred[0]['masks'])):  # Draws output segmentation according to the resulting score per-pixel
            msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
            scr = pred[0]['scores'][i].detach().cpu().numpy()
            if scr > 0.8:
                instance_color = generate_random_color()
                im2[:, :, 0][msk > 0.5] = instance_color[0]
                im2[:, :, 1][msk > 0.5] = instance_color[1]
                im2[:, :, 2][msk > 0.5] = instance_color[2]
        cv2.imwrite(outDir + "/" + pth, im2)  # Save output image
        print(pth)
print("Finish!")
