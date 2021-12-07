from scipy.ndimage.measurements import label

import torch 
import timeit

from models import Yolov3Model
from config_reader import CocoConfigReader
from utils.dataset import CocoImagePathFileDataset
from utils.dataloader import get_data_loader
from utils.utils import non_max_suppression

YOLO_CONFIG_PATH = "../cfg/yolov3.cfg"
COCO_CONFIG_PATH = "../cfg/coco.data"

yolo = Yolov3Model(YOLO_CONFIG_PATH)
yolo.load_weights()

yolo.eval()

data_loader = get_data_loader(COCO_CONFIG_PATH, CocoConfigReader,
        CocoImagePathFileDataset, mode="train" if yolo.training else "valid")

for image, bbox in data_loader:
    out = yolo(image)
    out = out.to("cuda")

    out = non_max_suppression(out)
    
    timer = timeit.Timer(lambda: non_max_suppression(out))
    nms_time = timer.timeit(number=1)

    print ("Time taken for nms ", nms_time)

    break

# print ("Finished the forward pass")
# print ("Shape of yolo heads", [x.shape for x in out])