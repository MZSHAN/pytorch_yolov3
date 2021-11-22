from scipy.ndimage.measurements import label

from models import Yolov3Model
from config_reader import CocoConfigReader
from utils.dataset import CocoImagePathFileDataset
from utils.dataloader import get_data_loader

YOLO_CONFIG_PATH = "../cfg/yolov3.cfg"
COCO_CONFIG_PATH = "../cfg/coco.data"

yolo = Yolov3Model(YOLO_CONFIG_PATH)

data_loader = get_data_loader(COCO_CONFIG_PATH, CocoConfigReader, 
        CocoImagePathFileDataset)

for image, bbox in data_loader:
    out = yolo(image)

print ("Finished the forward pass")
print ("Shape of yolo heads", [x.shape for x in out])