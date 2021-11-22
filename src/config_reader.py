from errors import ConfigFileIncorrectFormat


class ConfigReader:
    def __init__(self, config_path):
        self.config_path = config_path
    
    @property
    def config_path(self):
        return self._config_path
    
    @config_path.setter
    def config_path(self, config_path):
        if not isinstance(config_path, str):
            raise TypeError("The config path should be a string")
        self._config_path = config_path

    def parse_config(self):
        raise NotImplementedError


class YoloConfigReader(ConfigReader):
    def __init__(self, config_path):
        super().__init__(config_path)

    #TODO: Remove the extra line written before returning from function
    def parse_config(self):
        """
        Function to parse the config file into a list of modules. Each module is a dictionary with the parsed values
        """
        with open(self.config_path, "a") as f:
            f.write("\n") #Append a new line to file so code below is succint
        
        config_file = open(self.config_path, 'r')

        darknet_modules, detection_modules = [], []
        
        #modules_list initially parses darket feature extractor modules
        #When detection modules start, it points to detection_modules and 
        #parses them in detection_modules
        modules_list = darknet_modules
        current_module = {}
        
        for line in config_file:
            line = line.strip()
            if line and line[:4] == "####":
                modules_list = detection_modules
            elif not line or line[0] == "#": 
                if len(current_module) > 1:
                    modules_list.append(current_module)
                    current_module = {}
            elif line[0] == "[":
                line = line[1:].split("]")[0]
                line = line.strip()
                current_module["module_name"] = line
            else:
                key, value = line.split("=")
                if "," in value:
                    current_module[key.strip()] = [float(v) for v in value.split(",")]
                else:
                    try:
                        current_module[key.strip()] = float(value) #If it's numerical string, use it
                    except ValueError:
                        current_module[key.strip()] = value # else it is a word
        if len(current_module) > 1:
            modules_list.append(current_module)
        
        hyper_parameters, *darknet_modules = darknet_modules

        curr_head = []
        yolo_heads = []
        for i, mods in enumerate(detection_modules):
            curr_head.append(mods)
            if mods["module_name"] == "yolo":
                yolo_heads.append(curr_head)
                curr_head = []

        return hyper_parameters, darknet_modules, yolo_heads


class CocoConfigReader(ConfigReader):
    def __init__(self, config_path):
        super().__init__(config_path)
    
    def parse_config(self):
        coco_config_dict = {}

        with open(self.config_path, "r") as coco_config:
            for line in coco_config:
                if not line:
                    continue
                split_line = line.split("=")
                if len(split_line) != 2:
                    raise ConfigFileIncorrectFormat("Coco config line should have exactly one =")
                key, value = split_line
                coco_config_dict[key.strip()] = value.strip()
        
        return coco_config_dict