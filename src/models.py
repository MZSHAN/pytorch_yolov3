from collections import namedtuple

import torch
from torch import nn

# from builders import DarkNetBuilder, YoloHeadBuilder
import builders
from config_reader import YoloConfigReader


SkipConnection = namedtuple("SkipConnection", ["source"])
FPNLayer = namedtuple("FPNLayer", ["sources"])


class Yolov3Model(nn.Module):
    def __init__(self, config_path, channels=3):
        super().__init__()
        hyper_params, darknet_config, yolo_head_configs = \
            YoloConfigReader(config_path).parse_config()

        filter_sizes = [channels] #Maintains filter sizes for building modules
        self.yolo_modules = [] #Contains the yolo layers to be used in forward
        
        builders.DarkNetBuilder(darknet_config).build_model(
            filter_sizes,
            self.yolo_modules
        )               
        builders.YoloHeadBuilder(yolo_head_configs).build_model(
            filter_sizes, 
            self.yolo_modules
        )
        
    def forward(self, inp):
        """
            Loops over all the yolo_modules to generate outputs of each layer
            The layer outputs are stored in a list to be used for skip 
            connections FPN components
        """
        image_size = inp.shape[2]

        yolo_module_outputs, detection_grid_outputs = [], []

        #Builder has ensured that all modules here are supported
        for module in self.yolo_modules:
            if isinstance(module, list): #Convolutional module
                for component in module:
                    inp = component(inp)
            elif isinstance(module, SkipConnection):
                inp = yolo_module_outputs[module.source] + inp#Add to prev input
            elif isinstance(module, FPNLayer):
                source_layers = [yolo_module_outputs[s] 
                                    for s in  module.sources]
                inp = torch.cat(source_layers, 1) # stack along channel dim
            elif isinstance(module, YoloLayer):
                inp = module(inp, image_size)
                detection_grid_outputs.append(inp)
            else:
                inp = module(inp) #Currently only Upsample
            
            yolo_module_outputs.append(inp)
        
        # Each component of detection_grid_outputs has shape
        # (batch_size, num_anchors, grid_height, grid_width, 
        #  outputs_per_anchor)
        if self.training:
            return detection_grid_outputs
        else:
            return torch.cat(detection_grid_outputs, 1) # stack along achor dim
    
    def __repr__(self):
        return str(self.yolo_modules)
        

class YoloLayer(nn.Module):
    def __init__(self, anchors, n_classes):
        super().__init__()
        self.neurons_per_anchor = 5 + n_classes
        self.num_of_anchors = len(anchors)

        #Reshape anchors to calculate tuned anchor boxes in forward 
        # aids numpy broadcast multiplication 
        self.register_buffer("cell_anchors", 
            torch.Tensor(anchors).float().view(1, self.num_of_anchors, 1, 1, 2))
        self.grid = None
        
    def forward(self, inp, image_size):
        """
            Args:
                inp: Input Tensor
                        Shape: (batch_size, #anchors_per_grid * (#classes + 5)
                        , grid_width, grid_height)
        """
        batch_size, _, grid_height, grid_width = inp.shape
        #self.grid_stride implementation requires square images
        if grid_width != grid_height:
            ValueError("Image Width and Height mismatch." 
                "Pad input image with zeros") 
        self.grid_stride = image_size // grid_height

        #Reshape the input to have anchor specific outputs along last dim
        inp = (inp.view(batch_size, self.num_of_anchors, 
            self.neurons_per_anchor, grid_height, grid_width)
            .permute(0, 1, 3, 4, 2).contiguous())
        
        if not self.training:
            if not self.grid:
                self.grid = self._make_grid(grid_height, grid_width, self.num_of_anchors)

            #Calculate center of anchor box
            inp[..., 0:2] = (inp[...,0:2].sigmoid() + self.grid) * self.grid_stride
            #Calculate heigh and width of anchor boxes
            inp[..., 2:4] = torch.exp(inp[...,2:4]) * self.cell_anchors
            #Calculate class probabilities
            inp[..., 4:] = inp[...,4:].sigmoid()
            inp.reshape(batch_size, -1, self.neurons_per_anchor)
        
        return inp

    @staticmethod
    def _make_grid(grid_height, grid_width, num_anchors):
        row_nums, col_nums = torch.meshgrid(torch.arange(grid_height),
                                torch.arange( grid_width))
        #Make grid and reshape to ease calculation  of center of anchor boxes
        grid = torch.stack((row_nums, col_nums), 2).view(
                                        1, num_anchors, 1, 1, 2)
        return grid