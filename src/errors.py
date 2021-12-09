class ConfigFileIncorrectFormat(Exception):
     """Exception is raised when the config file is not in the correct format.
 
     Attributes:
         message -- explanation of the error
     """
 
     def __init__(self, message="Config file format is incorrect"):
         self.message = message
         super().__init__(self.message)
 
     def __str__(self):
         return self.message


class ImageReadError(Exception):
     def __init__(self, message="Error in reading Image"):
         self.message = message
         super().__init__(self.message)
 
     def __str__(self):
         return self.message


class LabelFileReadError(Exception):
     def __init__(self, message="Error in reading Image"):
         self.message = message
         super().__init__(self.message)
 
     def __str__(self):
         return self.message    


class WeightFileLoadError(Exception):
     def __init__(self, message="Error in loading weight from file"):
         self.message = message
         super().__init__(self.message)
 
     def __str__(self):
         return self.message    
