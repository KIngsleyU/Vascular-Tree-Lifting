# run training code

from torch.utils.data import Dataset
import glob
import numpy as np
import scipy.io as io
import os
import cv2
import PIL
from torch import Tensor, einsum
from image import ImageDataset
from ground_truth import GroundTruthDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.ndimage.morphology import distance_transform_edt as edt

## Loss function from https://github.com/JunMa11/SegLoss
## original called GDIceLossV2

class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                    print("using cuda")
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)

        input = flatten(softmax_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.smooth)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)
    
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class Encoder(nn.Module):
    def __init__(self, z_size):
        super(Encoder, self).__init__()
        self.z_size = z_size
        self.n_features_min = 64
        self.n_channel = 1
        self.batch_size = 1
        self.cube_len = 1

        self.conv1 = nn.Conv2d(self.n_channel, self.n_features_min, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(self.n_features_min)

        self.conv2 = nn.Conv2d(self.n_features_min, self.n_features_min * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.n_features_min * 2)

        self.conv3 = nn.Conv2d(self.n_features_min * 2, self.n_features_min * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.n_features_min * 4)

        self.conv4 = nn.Conv2d(self.n_features_min * 4, self.n_features_min * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.n_features_min * 8)

        self.conv5 = nn.Conv2d(self.n_features_min * 8, self.n_features_min * 16, 4, 1, 0)
        self.bn5 = nn.BatchNorm2d(self.n_features_min * 16)

        self.fc = nn.Linear(self.n_features_min * 16 * 9, self.z_size)

    def forward(self, input):
        batch_size = input.size(0)
        layer1 = F.leaky_relu(self.bn1(self.conv1(input)), 0.2)
        layer2 = F.leaky_relu(self.bn2(self.conv2(layer1)), 0.2)
        layer3 = F.leaky_relu(self.bn3(self.conv3(layer2)), 0.2)
        layer4 = F.leaky_relu(self.bn4(self.conv4(layer3)), 0.2)
        layer5 = F.leaky_relu(self.bn5(self.conv5(layer4)), 0.2)
        layer6 = layer5.view(batch_size, self.n_features_min * 16 * 9)
        layer6 = self.fc(layer6)
        return layer6


class Generator(nn.Module):
    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.z_size = z_size

        self.n_features_min = 64
        self.n_channel = 1
        self.batch_size = 1
        self.cube_len = 1

        # ConvTranspose3d
            # 1st parameter in_channel
            # 2nd parameter out_channel effects the 2nd dimension
            # 3rd parameter kernel
            # 4th parameter stride
            # 5th parameter padding

        self.conv1 = nn.ConvTranspose3d(self.z_size, self.n_features_min * 8, 3, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(self.n_features_min * 8)
        self.conv2 = nn.ConvTranspose3d(self.n_features_min * 8, self.n_features_min * 4, 4, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(self.n_features_min * 4)
        self.conv3 = nn.ConvTranspose3d(self.n_features_min * 4, self.n_features_min * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.n_features_min * 2)
        self.conv4 = nn.ConvTranspose3d(self.n_features_min * 2, self.n_features_min * 1, 4, 2, 0, bias=False)
        self.bn4 = nn.BatchNorm3d(self.n_features_min * 1)
        self.conv5 = nn.ConvTranspose3d(self.n_features_min * 1, self.n_channel, 4, 3, 1, bias=False)


    def forward(self, input):
        x = input.view(input.size(0), self.z_size, 1, 1, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = torch.sigmoid(x)

        return x

def compute_distances(output_numpy, gdtm):
  distance = torch.mul(output_numpy, gdtm)
  return distance

def compute_dtm_gt(img_gt):

    fg_dtm = edt(np.logical_not(img_gt))

    return fg_dtm

## Running the Deep learning
def main():
    Z_SIZE = 64
    Testdataset = ImageDataset()
    gtDataset = GroundTruthDataset()
    dataloader = torch.utils.data.DataLoader(Testdataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("using cuda")
    else:
        print("using CPU")

    encoder = Encoder(Z_SIZE).to(device)
    generator = Generator(Z_SIZE).to(device)

    Diceloss = GDiceLoss(softmax_helper, smooth=1e-2)
    lr_d = 0.0025
    lr_g = 0.0025
    alpha = 0.01 
    saveAlpha = alpha
    encoderOptim   = torch.optim.Adam(encoder.parameters(), lr=lr_d)
    generatorOptim   = torch.optim.Adam(generator.parameters(), lr=lr_g)
    distance_transforms_gt = []

    # check if output/progress directory already exists otherwise create it
    if not os.path.exists('output/progress'):
        print("Creating directory '/output/progress'...\n")
        os.makedirs('output/progress/')
    
    # check if output/models directory already exists otherwise create it   
    if not os.path.exists('output/models'):
        print("Creating directory '/output/models'...\n")
        os.makedirs('output/models/')

    print('Starting training...')


    for epochs in range(1,100):

        print("epoch: ", epochs)
        for idx, (file_name, input) in enumerate(dataloader):

            input = input.unsqueeze(0).float()
            image = input.to(device)
            x = encoder(image)

            expected = gtDataset.getItem(file_name[0]).to(device)

            output = generator(x)
            size = output.size()[3]
            output = output.reshape([size,size,size])

            outputflatten = flatten(output)
            expectedflatten = flatten(expected)

            diceLoss = Diceloss(outputflatten, expectedflatten)
            # compute distance maps 
            with torch.no_grad():
                if epochs == 1:
                    # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
                    gt_dtm_npy = compute_dtm_gt(expected.cpu().numpy())
                    gt_dtm_npy = torch.from_numpy(gt_dtm_npy).to(device)
                    distance_transforms_gt.append(gt_dtm_npy)          
                    print(idx)
                else:
                    gt_dtm_npy = distance_transforms_gt[idx]

            distances = compute_distances(output, gt_dtm_npy)
            loss_hd = torch.max(distances)

            if (loss_hd < 1):
                alpha = 1 - saveAlpha
            else:
                alpha = saveAlpha

            loss = alpha*(diceLoss) + (1 - alpha) * loss_hd
            encoderOptim.zero_grad()
            generatorOptim.zero_grad()
            loss.backward()
            encoderOptim.step()
            generatorOptim.step()

        if (epochs == 1):
            print(file_name)
            expected_shape = expected.cpu().detach().numpy()
            expectedIndicies = np.argwhere(expected_shape >= 1)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(expectedIndicies[:,0],expectedIndicies[:,1],expectedIndicies[:,2], cmap='gray')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim([0,101])
            ax.set_ylim([0,101])
            ax.set_zlim([0,101])
            name = "/content/drive/MyDrive/cmpt340/progress/groundtruth.png"
            plt.savefig(name)
            plt.close(fig)

        acc = (torch.argmax(output, 1) == expected).float().mean()
        print('epoch {}, loss {:.9f}, acc {:.5f}'.format(epochs, loss.item(), acc))

        output_shape = output.cpu().detach().numpy()
        # print(output.size)
        # print("expected size: ", output_shape.shape)
        outputIndicies = np.argwhere(output_shape >= 1)
        print(outputIndicies)

        # print("expected indicies shape: ", expectedIndicies.shape)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(outputIndicies[:,0],outputIndicies[:,1],outputIndicies[:,2], cmap='gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,101])
        ax.set_ylim([0,101])
        ax.set_zlim([0,101])
        name = "./output/progress/" + str(epochs) + ".png"
        plt.savefig(name)
        plt.close(fig)
        if (epochs % 10 == 0):
            ## Save models ##
            torch.save(encoder.state_dict(), "./output/models/encoder" + str(epochs) + ".pt")
            torch.save(generator.state_dict(), "./output/models/generator" + str(epochs) + ".pt")
    print('Training is now complete! Models have been saved.')
    torch.save(encoder.state_dict(), "./output/models/encoder_final.pt")
    torch.save(generator.state_dict(), "./output/models/generator_final.pt")
if __name__ == "__main__":
    main()
