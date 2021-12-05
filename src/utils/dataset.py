from pathlib import Path
import warnings
import numpy as np

from  PIL import Image
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

from errors import ImageReadError, LabelFileReadError


#Basic Implementation - read image, convert to numpy array, exchange axes,
#   make the image a square by padding with zeros, shift labels according
#   to make congruent with padded image
#TODO: Add augmentations using Albumentations, imgaug and pytorch transforms
class CocoImagePathFileDataset(Dataset):
    """
        Map style dataset to load COCO dataset from a file having image paths
    """
    def __init__(self, image_path_file):
        """
        Args:
            image_path_file: file has paths of all the images that are part of 
                the dataset
        """
        self.image_paths = self._load_image_paths(image_path_file)
        # Assume avg file string length = 100, utf8 for alphabets takes 1 byte
        # So each image_file path string is 100bytes
        # Max size of Coco Train2014 is ~81k
        # So max size of the image_paths list is 8100k = 8.1Mb
        # Dataset object creation will take some time
        # It is only done once per dataloader so it's fine

    def __len__(self):
        return len(self.image_paths)
    
    def _load_image_paths(self, image_path_file):
        if not isinstance(image_path_file, str):
            raise ValueError(f"The image_path_file should be a string but got a {type(image_path_file)}")
            
        if not Path(image_path_file).is_file():
            raise FileNotFoundError(f"The image path file does not exist at {image_path_file}")
        
        image_paths = []
        with open(image_path_file, "r") as image_locations:
            for image_path in image_locations:
                image_path = image_path.strip()
                try:
                    self._check_label_present(image_path)
                    image_paths.append(image_path)
                except FileNotFoundError as e:
                    #If just label absent, ignore. If dir incorrect, alert
                    if not Path(image_path).parent.is_dir():
                        raise FileNotFoundError(f"The image does not exist"
                         f"at {image_path}")
            
        return image_paths
    
    @staticmethod
    def _check_label_present(image_loc):
        if "/images/" not in image_loc:
            raise ValueError("Image path must have the folder \"images\"")
        
        label_file = CocoImagePathFileDataset._get_labelfile(image_loc)

        if not Path(label_file).is_file():
            raise FileNotFoundError(f"The label file for {image_loc}"
                        f" is not present at {label_file}")
    
    @staticmethod
    def _get_labelfile(image_loc):
        """
        Generates label file locations for the images on the go
        """
        #label file exists, checked in constructor
        parent_dir, training_image = image_loc.split("images/")
        label_file = parent_dir + "labels/" +training_image.split(".")[0] + ".txt"

        return label_file

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
    
        image_tensor, label_tensor = self._get_square_tensor_from_image(
            image_path)
        
        return image_tensor, label_tensor

    #TODO: Make this a transform
    @staticmethod
    def _get_square_tensor_from_image(image_path, target_height=416):
        """
        Function takes an image path as input, reads the image, pads it with
        zeros to make it a square of target_height and returns a tensor 
        representation
        It also generates a generates a transformed labels

        Args:
            image_path(str): path of the image file. File should exist
            target_height(int): height and width of resized image to be returned

        returns:
            torch tensor of transfored image, tensor of transformed labels
        """
        try:
            image_np = np.array(Image.open(image_path))    
        except Exception:
            raise ImageReadError(f"Could not be read image: {image_path}")
        
        height, width, _ = image_np.shape

        total_pad_len = abs(height - width)
        pad_before, pad_after = (total_pad_len // 2, total_pad_len -
            total_pad_len//2)

        pad_sequence = (((pad_before, pad_after), (0, 0), (0, 0))
                if height <= width else ((0, 0, (pad_before, pad_after), (0, 0))))

        pad_image_np = np.pad(image_np, pad_sequence, mode="constant", constant_values=128)
        pad_image_np = pad_image_np/255. #normalize

        target_shape = (target_height, target_height, 3)
        square_image_np = resize(pad_image_np, target_shape, mode="reflect")

        #torch tensor representation needs channels as first axis
        image_tensor = torch.from_numpy(np.transpose(square_image_np, (2, 0, 1)))

        #find the left and top padding to move center of labels
        pad_top, pad_left = pad_sequence[0][0], pad_sequence[1][0]
        
        label_path = CocoImagePathFileDataset._get_labelfile(image_path)
        label_tensor = CocoImagePathFileDataset._label_tensor_for_square_img(
            label_path, pad_top, pad_left, 
            image_np.shape[0:2], pad_image_np.shape[0:2])

        return image_tensor.float(), label_tensor.float()

    @staticmethod
    def _label_tensor_for_square_img(label_path, pad_top, pad_left,
            prev_size, pad_size):
        """
        Function takes a label_file with labels for an image
        It returns a tensor with lables that are adjusted for the square image
        Labels are in terms of fraction of the padded image

        Since the labels are in fractions, the padded image can be resized
         and scaled, and teh labels will remain the same

        label file contains class, center_x , center_y, width, height
        The last 4 coordinates in terms of fraction of original image

        Args:
            label_file(str) : The location of the label file
            pad_top (float) : The number of pixels padded to the top of image
            pad_left(float) : The number of pixels padded to the left of image
            prev_size(iterable) : Size of the unpadded image (height, width)
            new_size(iterable) : Size  of the resized image (height, width)

        returns:
            torch tensor with the label modified for padding and resizing
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                box_labels = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            raise LabelFileReadError(f"Error in reading {label_path}")

        prev_height, prev_width = prev_size
        pad_height, pad_width = pad_size

        #Convert xywh to xyxy - get unnormalized top left and 
        # bottom right corner
        x1 = (box_labels[:,1] - box_labels[:,3]/2) * prev_width
        x2 = (box_labels[:,1] + box_labels[:,3]/2) * prev_width
        y1 = (box_labels[:,2] - box_labels[:,4]/2) * prev_height
        y2 = (box_labels[:,2] + box_labels[:,4]/2) * prev_height

        #Get padding shifted corners
        x1 = x1 + pad_left
        x2 = x2 + pad_left
        y1 = y1 + pad_top
        y2 = y2 + pad_top

        #calcualte padding shifted center from corners, normalize 
        # by padded width
        box_labels[:,1] = ((x1 + x2) / 2) / pad_width
        box_labels[:,2] = ((y1 + y2) / 2) / pad_height

        #get fractional width and height : from unpadded to padded
        box_labels[:,3] *= prev_width / pad_width
        box_labels[:,4] *= prev_height/ pad_height

        tensor_box_labels = torch.from_numpy(box_labels)
        return tensor_box_labels