import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import numpy as np
import glob
from image import ImageTestDataset
from ground_truth import GroundTruthTestDataset
from train import Encoder, Generator
from chamferdist import ChamferDistance
import math
import sklearn
## Evaluate encoder and generator ##

# fromhttps://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdataset = ImageTestDataset()
    gtDataset = GroundTruthTestDataset()

    dataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    Z_SIZE = 64
    encoder = Encoder(Z_SIZE).to(device)
    generator = Generator(Z_SIZE).to(device)



    MSElossEvaluation = torch.nn.MSELoss()
    chamferdist = ChamferDistance()
    ioULoss = IoULoss()
    
    ## change to directory where you saved the models ##
    ## path for first model
    EncoderPath = './output/models/encoder10.pt'
    GeneratorPath ='./output/models/generator10.pt'

    encoder.load_state_dict(torch.load(EncoderPath))
    generator.load_state_dict(torch.load(GeneratorPath))

    RMSEtotalLoss = 0
    ChamferTotalLoss = 0
    IOUTotalLoss = 0.0

    with torch.no_grad():
        encoder.eval()
        generator.eval()
        for idx, (file_name, input)  in enumerate(dataloader):

            print('{}/{}'.format(idx + 1, len(dataloader)))
            expected = gtDataset.getItem(file_name[0]).to(device)
            # print(expected.shape)
            # print(file_name)
            input = input.unsqueeze(0).float()
            image = input.to(device)
            
            x = encoder(image)
            output = generator(x)
            size = output.size()[3]
            output = output.reshape([size,size,size])

            # print("RMSE loss")
            mse = MSElossEvaluation(output, expected)
            RMSEtotalLoss += math.sqrt(mse)

            output = output.cpu().detach().numpy()
            expected = expected.cpu().detach().numpy()

            outputIndicies = np.argwhere(output >= 1)
            expecetedIndicies = np.argwhere(expected >= 1)

            # print("chamfer loss")
            expecetedIndicies2 = np.expand_dims(expecetedIndicies, axis=0)
            outputIndicies2 = np.expand_dims(outputIndicies, axis=0)

            ChamferTotalLoss += chamferdist(torch.from_numpy(outputIndicies2).float(), torch.from_numpy(expecetedIndicies2).float()).cpu().detach().numpy()

            # print("IOU loss")
            size = max(expecetedIndicies.shape[0],outputIndicies.shape[0])
            
            if size > expecetedIndicies.shape[0]:
                addPadding = size - expecetedIndicies.shape[0]
                zeros = np.zeros([addPadding, 3])
                expecetedIndicies = np.append(expecetedIndicies, zeros, axis=0)
            else:
                addPadding = size - outputIndicies.shape[0]
                zeros = np.zeros([addPadding, 3])
                outputIndicies = np.append(outputIndicies, zeros, axis=0)

            IOUTotalLoss += ioULoss(torch.from_numpy(outputIndicies), torch.from_numpy(expecetedIndicies)).cpu().detach().numpy()

    print("Average RMSE loss for ", idx + 1, " with ", RMSEtotalLoss / (idx + 1))
    print("Average Chamfer loss for ", idx + 1, " with ", ChamferTotalLoss / (idx + 1))
    print("Average IoU loss for ", idx + 1, " with ", IOUTotalLoss / (idx + 1))

if __name__ == "__main__":
    test()
