import torch


#Tested
def non_max_suppression(anchor_outputs, object_thresh=0.5, iou_thresh=0.5):
    """
        Function implements non maximal supression of detections from anchors

        Funtion picks all anchor boxes which have an object with 
        probability > object_thresh
        It then continuously picks anchors boxes for a particular class with 
        max class probability and rejects all boxes of same class that have
        IOU > iou_thresh : as these were duplicate detections

        ****Function changes the reprentation of anchor_boxes
            (see args and return)
        
        Args:
        anchor_outputs(torch.tensor) : 
            shape  - (batch_size, total_anchors, outputs_per_anchor)
            tensor containing anchor outputs for all the anchors(of all grids)
                for the image
            anchor_outputs[:,:, 0:4] - center_x, center_y, height, width
            anchor_outputs[:,:,4] - objectness score of object in anchor box
            anchor_outputs[:,:,5:] - class probabilities of object in box
        object_thresh(float):
            objectness score threshold for detection to be an object
        iou_thresh(float):
            iou threshold to suppress duplicate detections in image

        returns:
        tensor containing (image_index, x1, y1, x2, y2, objectness score,
            class probability, class prediction ) where x1, y1, x2, y2
            are top left and bottom right corners of the bounding box
    """
    if len(anchor_outputs.shape) != 3:
        raise ValueError(f"Expected a 3D tensor but received " 
            f"{len(anchor_outputs.shape)}D tensor")

    total_detections = torch.sum(anchor_outputs[:, :, 4] >= object_thresh)
    if not total_detections: 
        # no images boxes with detections
        return 
    
    # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
    anchor_outputs[...,:4] = get_bbox_corners(anchor_outputs[...,:4])

    nms_output = []

    #TODO : Parallize batch computations with CUDA(Streams?)
    # discuss.pytorch.org/t/batch-non-maximum-suppression-on-the-gpu/34210    
    for image_index, image_preds in enumerate(anchor_outputs):
        object_indices = image_preds[:,4] >= object_thresh        
        if not torch.sum(object_indices): #if no object, skip image
            continue

        #ignore boxes with no object
        image_preds = image_preds[object_indices]

        # Get max class probability and predicted class for all bounding boxes
        class_probs, class_pred = torch.max(image_preds[:, 5:],
                                    axis=1, keepdim=True)

        image_detections = torch.cat((image_preds[:, :5], 
                            class_probs.float(), class_pred.float()), axis=1)
        #get all unique object classes in image
        clasess_present = torch.unique(class_pred)

        nms_image_detections = []
        for curr_class in clasess_present:
            #get all detections of a class
            class_detections = image_detections[image_detections[:,-1] == curr_class]

            #Sort detections with decreasing probability before suppression
            # This is coz we want the highest prob detection to be retained and low
            # prob detections to be suppressed
            _, sort_indices = torch.sort(class_detections[:, 4], descending=True)
            class_detections = class_detections[sort_indices]

            nms_class_detections = []

            while len(class_detections):
                #picks high prob detection
                nms_class_detections.append(class_detections[0])

                #final class detection considered, edge case
                if len(class_detections) == 1:
                    break

                #Suppress classes having iou > iou_thresh 
                class_detections = class_detections[
                    get_bbox_iou(nms_class_detections[-1].unsqueeze(0), 
                    class_detections) < iou_thresh
                ]

            #accumulate image level detections
            nms_image_detections += nms_class_detections

        #convert to tensor
        nms_image_detections = torch.stack(nms_image_detections)
        
        # get image index tensor of same shape on same device
        image_index_tensor = nms_image_detections.new_empty(
                                (nms_image_detections.shape[0], 1)
                                ).fill_(image_index)
        
        nms_image_detections = torch.cat((image_index_tensor,
                                nms_image_detections), axis=1)
        nms_output.append(nms_image_detections)
        
    nms_output = torch.stack(nms_output) if nms_output else None

    return nms_output


def get_bbox_iou(bbox1, bbox2):
    """
    Function to calculate intersection over union for two bounding boxes

    Args:
        bbox1(tensor): shape(:, 4) 
            top left and bottom right co ordinates of the bouding box
        bbox2(tensor): shape(:, 4) 
            top left and bottom right co ordinates of the bouding box
    returns:
        if bbox1 and bbox2 have same number of boxes
            returns pairwise iou of bbox1 and bbox2
        if bbox1 is single box and bbox2 is a tensor of boxes,
            returns iou of bbox1 with all boxes of bbox2
        Similar for if bbox2 is a single box and bbox1 is a tensor of boxes
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:,1], bbox1[:,2], bbox1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:,1], bbox2[:,2], bbox2[:,3]

    # length * breadth
    bbox1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    bbox2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    #get intersection coordinates if it exists
    intersection_x1 = torch.max(b1_x1, b2_x1)
    intersection_y1 = torch.max(b1_y1, b2_y1)
    intersection_x2 = torch.min(b1_x2, b2_x2)
    intersection_y2 = torch.min(b1_y2, b2_y2)

    #calculate intersection area, zero if no intersection
    intersection_area = (
        torch.clamp(intersection_x2 - intersection_x1 + 1, min=0) * 
        torch.clamp(intersection_y2 - intersection_y1 + 1, min=0)
    )

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou


def get_bbox_corners(center_bbox_repr):
    """
    Function converts bounding box representation from 
        (center_x, center_y, height, width) to (x1, y1, x2, y2)
    
    Args:
        center_box_repr(tensor) - shape (..., 4)
    returns:
        tensor - shape (..., 4)
    """
    if center_bbox_repr.shape[-1] != 4:
        raise ValueError("The center representation of bounding box"
            f"must have 4 values. Received {center_bbox_repr.shape[-1]}")

    center_x = center_bbox_repr[...,0]
    center_y = center_bbox_repr[...,1]
    half_height = center_bbox_repr[...,2] / 2
    half_width = center_bbox_repr[...,3] / 2

    #get new placeholder tensor on same device of same dtype
    box_preds = center_bbox_repr.new_empty(center_bbox_repr.shape) 
    box_preds[...,0] =  center_x - half_height
    box_preds[...,1] = center_y - half_width
    box_preds[...,2] = center_x + half_height
    box_preds[...,3] = center_y + half_width 

    return box_preds


def load_classes(classes_file):
    """
    Function to load names of classification classes

    Args:
        classes_file(str) - path to file that contains class names
    returns:
        array containing names of classes at the corresponding cllass index
    """
    fp = open(classes_file, "r")
    class_names = fp.read().split("\n")[:-1]
    return class_names


def convert_center_to_corner_repr(box_labels):
    """
    Function to convert bounding box representatation from  (center_x, 
        center_y, height, width) to 2 corner format (x1, y1, x2, y2)
    
    Args:
        box_labels(tensor) - shape(:, 4)
            contains anchor box representation in (center_x, 
            center_y, height, width) format
    returns: 
        tensor - shape(:,4)
            anchor boxes in (x1, y1, x2, y2) format
    """
    x1 = (box_labels[:,0] - box_labels[:,2]/2).reshape(-1, 1)
    x2 = (box_labels[:,0] + box_labels[:,2]/2).reshape(-1, 1) 
    y1 = (box_labels[:,1] - box_labels[:,3]/2).reshape(-1, 1) 
    y2 = (box_labels[:,1] + box_labels[:,3]/2).reshape(-1, 1) 

    box_labels = torch.cat((x1, y1, x2, y2), axis = 1)
    
    return box_labels


def convert_corner_to_pyplot_repr(box_labels):
    """
    Function to convert two corner representation of anchor boxes
        1 corner, height and width
    
    Pyplot patches.rectangle requires bbox representation in this
    format

    Args:
        box_labels(tensor) - shape(:, 4)
            contains anchor box representation in (x1, y1, x2, y2) format
        
    returns:
        anchor box representation in top_left_x, top_left_y, height, width
    """
    x = (box_labels[:,0]).reshape(-1,1)
    y = (box_labels[:,1]).reshape(-1,1)
    height = (box_labels[:,2] - box_labels[:,0]).reshape(-1, 1)
    width = (box_labels[:,3] - box_labels[:,1]).reshape(-1, 1)

    return torch.cat((x, y, height, width), axis = 1)


def convert_corner_to_center_repr(box_labels):
    """
    Function to convert bounding box representatation from  2 corner format
     (x1, y1, x2, y2) to (center_x, center_y, height, width)
    
    Args:
        box_labels(tensor) - shape(:, 4)
            contains anchor box representation in (x1, y1, x2, y2) format
    returns: 
        tensor - shape(:,4)
            anchor boxes in (center_x, center_y, height, width) format
    """    
    x = ((box_labels[:,0] + box_labels[:, 2]) / 2).reshape(-1, 1)
    y = ((box_labels[:,1] + box_labels[:, 3]) / 2).reshape(-1, 1)
    height = (box_labels[:,2] - box_labels[:,0]).reshape(-1, 1)
    width = (box_labels[:,3] - box_labels[:,1]).reshape(-1, 1)

    return torch.cat((x, y, height, width), axis = 1)


def get_four_corners_from_2_corners(x1, y1, x2, y2):
    """
    Function returns all corners of a bounding box given 2 corners

    Args:
        x1, y1, x3, y2 (int) - top left and bottom right corners of
            box
        returns
            list containing all corners of box. 
    """
    return [x1, y1, x1, y2, x2, y2, x2, y1]