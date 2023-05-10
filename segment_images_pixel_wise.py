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
maskDir = "mask_input"
outDir = "output" 	    # Directory for output segmentations. Directory must already exist.
maskColor = [255, 255, 255]  # Mask color values 0-255. 128R, 64G, 128B to be widely used for roads.
bgOn = False 		    # Toggle for including original image as background in output. Turn to False for mask only output
checkpoint = "1000_no_style.torch" # Trained checkpoint file to use.


### Mask R-CNN model initialization, evaluation and output, lightly modified code courtesy of Sagi Eppel:
### https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
model.load_state_dict(torch.load(checkpoint))
model.to(device) # move model to the right device (GPU or CPU)
model.eval()


im_list = os.listdir(imDir)
mask_list = os.listdir(maskDir)
sim_avg = 0
for pth in range(len(im_list)): # Iterates through images
    images = None
    images = cv2.imread(imDir + "/" + im_list[pth])
    mask = cv2.imread(maskDir + "/" + mask_list[pth], cv2.IMREAD_GRAYSCALE)
    if images is not None:
        images = cv2.resize(images, imageSize, cv2.INTER_LINEAR) # Resize image
        mask = cv2.resize(mask, imageSize, cv2.INTER_LINEAR) # Resize image
        images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0) # Preparation for evaluation
        images = images.swapaxes(1, 3).swapaxes(2, 3)
        images = list(image.to(device) for image in images)

        threshold_value = 1
        max_value = 255
#         mask = cv2.resize(mask, imageSize, cv2.INTER_LINEAR)  # Resize mask to match the target imageSize
        ret, binary_img = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)


        with torch.no_grad():
            pred = model(images)

        im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)

        if (bgOn): # Draws background or not depending on the boolean "bgOn"
            im2 = im.copy()
        else:
            im2 = np.zeros((imageSize[1], imageSize[0], 3), dtype = im.dtype)

        for i in range(len(pred[0]['masks'])): # Draws output segmentation according to the resulting score per-pixel
            msk=pred[0]['masks'][i, 0].detach().cpu().numpy()
            scr=pred[0]['scores'][i].detach().cpu().numpy()
            if scr>0.8 :
                im2[:, :, 0][msk > 0.5] = maskColor[0]
                im2[:, :, 1][msk > 0.5] = maskColor[1]
                im2[:, :, 2][msk > 0.5] = maskColor[2]
#         cv2.imwrite(outDir + "/" + im_list[pth], im2) # Save output image
#         cv2.imwrite(outDir + "/" + mask_list[pth], binary_img)
        print(im_list[pth])

        ret, binary_im2 = cv2.threshold(im2[:, :, 0], threshold_value, max_value, cv2.THRESH_BINARY)
        print("binary_im2 shape:", binary_im2.shape)
        print("binary_img shape:", binary_img.shape)

        diff = cv2.compare(binary_im2, binary_img, cv2.CMP_NE)

        # Count the number of non-zero pixels in the difference image
        non_zero_pixels = cv2.countNonZero(diff)

        # Calculate the percentage of similarity
        similarity = 100 - (non_zero_pixels / (binary_im2.size) * 100)
        sim_avg+=similarity
        print("Similarity: "+str(similarity))
sim_avg = sim_avg/len(im_list)
print("Average similarity: "+str(sim_avg))

print("Finish!")
