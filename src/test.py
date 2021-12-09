import random
import timeit
from PIL import Image

from models import Yolov3Model
from config_reader import CocoConfigReader
from utils.dataset import CocoImagePathFileDataset
from utils.dataloader import get_data_loader
from utils.utils import (convert_corner_to_pyplot_repr,
    non_max_suppression, load_classes)
    
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


YOLO_CONFIG_PATH = "../cfg/yolov3.cfg"
COCO_CONFIG_PATH = "../cfg/coco.data"


def plot_detections_on_image(detections, image):
    #Ignore image index
    detections = detections[:,1:]

    #load class names
    classes=load_classes("/home/ubuntu/workspace/pytorch_yolov3/data/coco.names")

    img = image.permute(1, 2, 0).numpy()
    img = Image.fromarray(np.uint8(img*255))

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, 30)]

    unique_labels = torch.unique(detections[:, -1]).cpu()
    n_cls_preds = unique_labels.shape[0]
    bbox_colors = random.sample(colors, n_cls_preds)

    for prediction in detections:
        x1, y1, h, w = convert_corner_to_pyplot_repr(
                        prediction[:4].unsqueeze(0)).squeeze()
        class_prob = prediction[-2]
        pred_class = prediction[-1]

        color = bbox_colors[int(np.where(unique_labels == int(pred_class))[0])]

        bbox = patches.Rectangle((x1, y1), h, w, linewidth=2,
                                    edgecolor=color,
                                    facecolor="none")
        # Add the bbox to the image
        ax.add_patch(bbox)
        # Add class with probability
        plt.text(x1, y1, s="P(" + classes[int(pred_class)] +f")={class_prob:.2f}", 
                color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})

    plt.axis('off') #remove axes
    plt.gca().xaxis.set_major_locator(NullLocator())#remove axis markings
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('../inference_test.png' , bbox_inches='tight', pad_inches=0.0)
    plt.close()


if __name__ == "__main__":
    yolo = Yolov3Model(YOLO_CONFIG_PATH)
    yolo.load_weights()
    yolo.eval()

    data_loader = get_data_loader(COCO_CONFIG_PATH, CocoConfigReader,
            CocoImagePathFileDataset, mode="valid")

    for i, (image, _) in enumerate(data_loader):
        out = yolo(image)
        out = out.to("cuda")

        #Also changes (center_x, center_y, x, y) to (x1, y1, x2, y2)
        detections = non_max_suppression(out, object_thresh=0.7)
        
        plot_detections_on_image(detections[0], image[0])

        print ("Image generated")
        break