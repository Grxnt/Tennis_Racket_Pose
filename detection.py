import numpy as np
import torch
import pycocotools
import cv2

from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

#The first thing to do is to hard code images to test... We'll add in args later

# Load Image to process
tennis = Image.open("images/control.jpg")
tennis_offset = Image.open("images/40_percent.jpg")
# Convert to torch tensor
tennis_tensor = pil_to_tensor(tennis)
tennis_offset_tensor = pil_to_tensor(tennis_offset)
# Add batch dimension
tennis_tensor_b = tennis_tensor.unsqueeze(dim=0)
tennis_offset_b = tennis_offset_tensor.unsqueeze(dim=0)
# Convert to float
tennis_tensor_float = tennis_tensor_b/255.0
tennis_offset_float = tennis_offset_b/255.0
# Load Model
Weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=Weights, box_score_thresh=0.9)
model.eval(); ## Setting Model for Evaluation/Prediction

# Detect Objects
tennis_preds = model(tennis_tensor_float)
offset_preds = model(tennis_offset_float)

print(tennis_preds)

tennis_preds[0]["boxes"]  = tennis_preds[0]["boxes"][tennis_preds[0]["scores"] > 0.8]
tennis_preds[0]["labels"] = tennis_preds[0]["labels"][tennis_preds[0]["scores"] > 0.8]
tennis_preds[0]["scores"] = tennis_preds[0]["scores"][tennis_preds[0]["scores"] > 0.8]

offset_preds[0]["boxes"]  = offset_preds[0]["boxes"][offset_preds[0]["scores"] > 0.8]
offset_preds[0]["labels"] = offset_preds[0]["labels"][offset_preds[0]["scores"] > 0.8]
offset_preds[0]["scores"] = offset_preds[0]["scores"][offset_preds[0]["scores"] > 0.8]

# Stack the two tensors together along a new dimension
predictions = []
predictions.append(tennis_preds)
predictions.append(offset_preds)

# The for loop below takes the bounding boxes of the detected tennis rackets and combines them into a single list. 
# The first item in the list will always be the control racket, as it will always be concatenated to the front of the
# list.

print(tennis_preds)

first_racket = True
point_labels = []
for pred in predictions:
    idx = 0
    for i in pred[0]["labels"]:
        if(i == torch.tensor(43)):
            # Convert box to 2D tensor so that it can be read by "draw_bounding_boxes"
            bb_resize = pred[0]["boxes"][idx].resize(1,4)
            # Add box to the back of the list
            point_labels.append(pred[0]["boxes"][idx].tolist())
            # Draw box on image
            tennis_tensor = draw_bounding_boxes(tennis_tensor, boxes=bb_resize, labels=["Racket"], colors="red", width=2, font_size=20)
            
        idx = idx + 1

print(point_labels)

# Use the bound box given to use by model to isolate tennis racket (Control)
bb0 = point_labels[0]
tennis_np = np.asarray(tennis)
tennis_snip = tennis_np[int(bb0[1]):int(bb0[3]),int(bb0[0]):int(bb0[2])]

# Use the bound box given to use by model to isolate tennis racket (Experiment)
bb0 = point_labels[1]
tennis_np = np.asarray(tennis_offset)
offset_snip = tennis_np[int(bb0[1]):int(bb0[3]),int(bb0[0]):int(bb0[2])]


#final_image = to_pil_image(tennis_tensor)

#final_image.show()

#Thing todo...

# Need to get this perspective shift working... and the best way to do this is to write some functions
# 1) Need to write a function that returns the coordinates of a tennis racket from a given image... nn will definitely help here
# 2) Need to use these coordinates to transform the image back to the size of the control image --> warpPerspective/getPerspectiveTransform
# 3) Use the transform matrix to calculate angles --> RQDecomp3x3() [https://docs.opencv.org/4.1.1/d9/d0c/group__calib3d.html#ga1aaacb6224ec7b99d34866f8f9baac83]