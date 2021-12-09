from collections import namedtuple
from typing import List

import numpy as np
import torch
from torch import nn

# from builders import DarkNetBuilder, YoloHeadBuilder
import builders
from config_reader import YoloConfigReader
from errors import WeightFileLoadError


# SkipConnection and FPNLayer classes are hacky
# Made them into nn modules to include in nn.ModuleList
class SkipConnection(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source


class FPNLayer(nn.Module):
    def __init__(self, sources):
        super().__init__()
        self.sources = sources


class Yolov3Model(nn.Module):
    def __init__(self, config_path, channels=3):
        super().__init__()
        hyper_params, darknet_config, yolo_head_configs = \
            YoloConfigReader(config_path).parse_config()

        filter_sizes = [channels] #Maintains filter sizes for building modules
        # Will contain the yolo layers to be used in forward
        self.yolo_modules = nn.ModuleList()
        
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
            if isinstance(module, nn.ModuleList): #Convolutional module
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


    #TODO: Remove location hardcoding - use pathlib
    def load_weights(self, weights_file="../weights/yolov3.weights"):
        """
        Function loads the downloaded weigts from pjreddie to the yolov3 net
        Logic follows:
        https://github.com/pjreddie/darknet/blob/f8f67cd461f20b71fd2ae1a46d04e99ed9f9f51d/src/parser.c#L1218

        Args:
            weights_file(str): location of the weights file
        """
        weights_fp = open(weights_file, "rb")
        #Advance file pointer as first 20 bytes are header vals
        np.fromfile(weights_fp, dtype=np.int32, count=5)

        temp = 0
        for module in self.yolo_modules:
            #Only convolutional component have trainable weights
            #All other layers are either reshaping or redirecting input
            if not isinstance(module, nn.ModuleList):
                continue
            
            #Only executed for convolutional modules
            if not module or not isinstance(module[0], torch.nn.Conv2d):
                raise ValueError("Weights should be loaded for,"
                " Convolutional module")

            with_batch_norm = False
            try:
                if isinstance(module[1], torch.nn.BatchNorm2d):
                    with_batch_norm = True
            except IndexError:
                pass #module only contains convolution, do nothing
            
            conv_layer = module[0]
            if with_batch_norm:
                batch_norm_layer = module[1]
                num_biases = batch_norm_layer.bias.numel()
                num_weights = batch_norm_layer.weight.numel()
                num_run_mean = batch_norm_layer.running_mean.numel()
                num_run_var = batch_norm_layer.running_var.numel()

                biases = self._torch_tensor_from_file_pointer(weights_fp, num_biases)
                weights = self._torch_tensor_from_file_pointer(weights_fp, num_weights)
                run_mean = self._torch_tensor_from_file_pointer(weights_fp, num_run_mean)
                run_var = self._torch_tensor_from_file_pointer(weights_fp, num_run_var)

                batch_norm_layer.bias.data.copy_(
                    biases.view_as(batch_norm_layer.bias))
                batch_norm_layer.weight.data.copy_(
                    weights.view_as(batch_norm_layer.weight))
                batch_norm_layer.running_mean.data.copy_(
                    run_mean.view_as(batch_norm_layer.running_mean))
                batch_norm_layer.running_var.data.copy_(
                    run_var.view_as(batch_norm_layer.running_var))
            else:
                #If no batch norm layer, convolutional layer has(requires) bias
                num_biases = conv_layer.bias.numel()
                biases = self._torch_tensor_from_file_pointer(weights_fp, num_biases)
                conv_layer.bias.data.copy_(
                    biases.view_as(conv_layer.bias))
            
            #Load convolutional weights
            num_weights = conv_layer.weight.numel()
            weights = self._torch_tensor_from_file_pointer(weights_fp, num_weights)
            conv_layer.weight.data.copy_(weights.view_as(conv_layer.weight))

        print ("Finished loading Yolov3 weights")  

    @staticmethod
    def _torch_tensor_from_file_pointer(file_pointer, count, dtype=np.float32):
        """
        Method to load a torch tensor of a given size from a file pointer of weight file

        Use Function to iteratively load weights without loading entire file
         in memory

        Args:
        file_pointer(_io.TextIOWrapper): File pointer to weight file
        count(int) : Number of elements to be read from file
        dtype(np.dtype) : type of elements to be read from file
        """
        np_array = np.fromfile(file_pointer, dtype=dtype, count=count)
        if np_array.size == 0:
            raise WeightFileLoadError(f"No more weights to load from" 
                f"file {file_pointer.name}")
        return torch.from_numpy(np_array)

        
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
            if self.grid is None:
                self.grid = self._make_grid(grid_height, grid_width)

            #Calculate center of anchor box
            inp[..., 0:2] = (inp[...,0:2].sigmoid() + self.grid) * self.grid_stride
            #Calculate heigh and width of anchor boxes
            inp[..., 2:4] = torch.exp(inp[...,2:4]) * self.cell_anchors
            #Calculate class probabilities
            inp[..., 4:] = inp[...,4:].sigmoid()
            inp = inp.view(batch_size, -1, self.neurons_per_anchor)
        
        return inp

    @staticmethod
    def _make_grid(grid_height, grid_width):
        col_nums, row_nums = torch.meshgrid(torch.arange(grid_height),
                                torch.arange( grid_width))
        #Make grid and reshape to ease calculation  of center of anchor boxes
        grid = torch.stack((row_nums, col_nums), 2).view(
                                        1, 1, grid_height, grid_width, 2)
        return grid