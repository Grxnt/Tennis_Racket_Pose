import numpy as np
import torch
#import pycocotools
import cv2

from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

#The first thing to do is to hard code images to test... We'll add in args later

def detect(filename):
    # Load Image to process
    tennis = Image.open(filename)
    #Get Image size to compute if we need to rotate
    rotate = False
    [width, height] = tennis.size
    if height > width:
        rotate = True
    # Convert to torch tensor
    tennis_tensor = pil_to_tensor(tennis)
    # Add batch dimension
    tennis_tensor_b = tennis_tensor.unsqueeze(dim=0)
    # Convert to float
    tennis_tensor_float = tennis_tensor_b/255.0
    # Load Model
    Weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=Weights, box_score_thresh=0.9)
    model.eval(); ## Setting Model for Evaluation/Prediction
    
    # Detect Objects
    tennis_preds = model(tennis_tensor_float)
    
    print(tennis_preds)
    
    tennis_preds[0]["boxes"]  = tennis_preds[0]["boxes"][tennis_preds[0]["labels"] == 43]
    tennis_preds[0]["masks"]  = tennis_preds[0]["masks"][tennis_preds[0]["labels"] == 43]
    tennis_preds[0]["scores"] = tennis_preds[0]["scores"][tennis_preds[0]["labels"] == 43]
    tennis_preds[0]["labels"] = tennis_preds[0]["labels"][tennis_preds[0]["labels"] == 43]
    
    tennis_tensor = draw_bounding_boxes(tennis_tensor, boxes=tennis_preds[0]["boxes"], labels=["Racket"], colors="red", width=2, font_size=20)
    
    final_image_pil = to_pil_image(tennis_preds[0]["masks"][0])
    final_image_np = np.array(final_image_pil)
    if rotate:
        final_image_np = np.rot90(final_image_np,3)
    
    cv2.imwrite('pog.jpg', final_image_np)
    final_image_np = cv2.imread('pog.jpg')
    final_image_np = cv2.cvtColor(final_image_np, cv2.COLOR_BGR2GRAY)
    return final_image_np

#Thing todo...

# Need to get this perspective shift working... and the best way to do this is to write some functions
# 1) Need to write a function that returns the coordinates of a tennis racket from a given image... nn will definitely help here
# 2) Need to use these coordinates to transform the image back to the size of the control image --> warpPerspective/getPerspectiveTransform
# 3) Use the transform matrix to calculate angles --> RQDecomp3x3() [https://docs.opencv.org/4.1.1/d9/d0c/group__calib3d.html#ga1aaacb6224ec7b99d34866f8f9baac83]