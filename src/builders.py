import torch
from torch import nn
from torch.nn.modules.activation import LeakyReLU

import models
from errors import ConfigFileIncorrectFormat

MODULE_MAP = {
        "convolutional" : "conv_builder",
        "shortcut"      : "skip_connection_builder",
        "yolo"          : "yolo_builder",
        "route"         : "fpn_layer_builder",
        "upsample"      : "upsample_builder"
    }

#TODO: include all torch.nn activations in this map
ACTIVATION_MAP = {
    "leaky" : nn.LeakyReLU
}

   
class Builder:
    def __init__(self, components):
        self.components = components

    @property
    def components(self): 
        return self._components
    
    @components.setter
    def components(self, components):
        if not isinstance(components, list):
            raise TypeError("Darknet components should be a list")
        self._components = components
        
    def build_model(self, filters, modules):
        if not isinstance(filters, list) and not isinstance(modules, list):
            raise TypeError("Filters and modules shoud be lists")
        
        for component in self.components:
            try:
                module_builder = getattr(self, MODULE_MAP[component["module_name"]])
            except AttributeError as e:
                unavail_module = MODULE_MAP[component["module_name"]]
                raise AttributeError((f"Check module name {unavail_module}"
                    f"It is currently not supported"))

            curr_module = module_builder(component, filters)
            modules.append(curr_module)
        

    @staticmethod
    def conv_builder(conv_componet, filters):
        conv_module = nn.ModuleList()
        try:
            arg_dict = {"in_channels":filters[-1]}
            arg_dict["out_channels"] = int(conv_componet["filters"])
            arg_dict["kernel_size"] = int(conv_componet["size"])
            arg_dict["stride"] = int(conv_componet["stride"])
            # Yolo config has pad of 1 for (1x1) convolutions 
            #   which causes shape mismatch - infer pad from kernel size
            arg_dict["padding"] = arg_dict["kernel_size"] // 2 
            arg_dict["bias"] = not conv_componet.get("batch_normalize", 0)

            conv_module.append(nn.Conv2d(**arg_dict))
            if not arg_dict["bias"]:
                conv_module.append(nn.BatchNorm2d(arg_dict["out_channels"]))

            activation_fn = ACTIVATION_MAP.get(conv_componet["activation"], None)
            if activation_fn:
                #TODO: Check if leaky slope has to be changed
                if activation_fn == LeakyReLU:
                    conv_module.append(activation_fn(0.1))
                else:
                    conv_module.append(activation_fn())

            filters.append(arg_dict["out_channels"])
            return conv_module

        except KeyError as e:
            raise ConfigFileIncorrectFormat("Convolutional component in",
                "config should have filters, size, stride, pad and activation")

    @staticmethod
    def skip_connection_builder(skip_component, filters):
        filters.append(filters[-1])
        skip_module = models.SkipConnection(int(skip_component["from"]))
        return skip_module
    
    @staticmethod
    def yolo_builder(yolo_component, filters):
        all_anchors = list(map(int, yolo_component["anchors"]))
        anchor_masks = list(map(int,yolo_component["mask"]))

        # all_anchors = list(map(int, yolo_component["anchors"].split(",")))
        # anchor_masks = list(map(int, yolo_component["mask"].split(",")))

        anchors = [(all_anchors[2*i], all_anchors[2*i+1]) for i in anchor_masks]
        n_classes = int(yolo_component["classes"])

        # Since this output layer, won't be used 
        filters.append(filters[-1]) 

        return models.YoloLayer(anchors, n_classes)

    @staticmethod
    def fpn_layer_builder(fpn_component, filters):
        #if not a list, it will be float or int
        if not isinstance(fpn_component["layers"], list):
            fpn_component["layers"] = [fpn_component["layers"]]
        
        fpn_layer_module = models.FPNLayer([int(x) for x in 
                fpn_component["layers"]])
        
        curr_out_channel = 0
        for layer in fpn_layer_module.sources:
            curr_out_channel += filters[layer]
        filters.append(curr_out_channel)
        return fpn_layer_module
    
    @staticmethod
    def upsample_builder(upsample_component, filters):
        filters.append(filters[-1]) # Feature map enlarges, but channels same
        return nn.Upsample(scale_factor=upsample_component["stride"], 
                mode="nearest")


class DarkNetBuilder(Builder):
    def __init__(self, components):
        super().__init__(components)

    
class YoloHeadBuilder(Builder):
    """
    Class to build Yolo Heads from list of yolo-head components

    Yolohead takes the features from the darknet feature extractor, transforms
    the features(convolutions) and finally passes them through a yolo layer

    build_model function takes a list of modules and appends it with  modules 
    constructed from yolo-head components
    """
    def __init__(self, components):
        """
        Constructor for the Yolohead builder
        Args:
        compenents(list): List of yolo head component
        
            Each yolo head component is a list of possible module dictionaries
            Module dictioaries contain information about the torch nn
        """
        super().__init__(components)
    
    def build_model(self, filters, modules):
        all_yolo_heads = self.components
        for yolo_head in all_yolo_heads:
            if not isinstance(yolo_head, list):
                raise TypeError("YoloHeadBuilder object",
                    "should have a list of all yolo head components")
            self.components = yolo_head #parent class parses this var
            super().build_model(filters, modules)