import torch

from models import Yolov3Model

config_path = "../cfg/yolov3.cfg"
yolo = Yolov3Model(config_path)


inp = torch.ones((1, 3, 416, 416))
out = yolo(inp)

print ("Finished the forward pass")
print ("Shape of yolo heads", [x.shape for x in out])