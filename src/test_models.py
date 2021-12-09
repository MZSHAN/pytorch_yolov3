from models import YoloLayer

import torch

def test_yolo_layer():
    #(height, width)
    anchors = [(6, 2), (3, 5)]
    yolo_layer = YoloLayer(anchors, n_classes=2)
    yolo_layer.eval()

    # 2X2 grid with 7 channels for each anchor
    input_tensor = torch.ones((1, 14, 2, 2))
    #set width, height and center neurons as zero
    input_tensor[:,0:4,:,:] = 0
    input_tensor[:,7:11,:,:] = 0

    output_tensor = yolo_layer(input_tensor, image_size=100)

    assert(torch.tensor(output_tensor.shape).tolist() == [1, 8, 7])
    print ("Yolo layer output shape is as expected")

    expected_tensor = torch.ones((8,7))
    #Centers of 2X2 grid on 100 length image. center since sigmoid(0) = 0.5
    expected_tensor[0, 0:2] = expected_tensor[4,0:2] = torch.tensor([25,25])
    expected_tensor[1, 0:2] = expected_tensor[5,0:2] = torch.tensor([75,25])
    expected_tensor[2, 0:2] = expected_tensor[6,0:2] = torch.tensor([25,75])
    expected_tensor[3, 0:2] = expected_tensor[7,0:2] = torch.tensor([75,75])

    expected_tensor[:4, 2:4] = torch.tensor([6, 2])
    expected_tensor[4:, 2:4] = torch.tensor([3, 5])
    
    expected_tensor[:,4:] = expected_tensor[:,4:].sigmoid()
    expected_tensor = expected_tensor.unsqueeze(0)

    assert(torch.equal(output_tensor, expected_tensor))
    print ("Yolo layer works correctly")


if __name__ == "__main__":
    test_yolo_layer()