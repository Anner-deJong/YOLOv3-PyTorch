# This file contains helper functions and layers for the Yolo v3 network

import torch
import torch.nn as nn
import numpy as np

# FROM TUTORIAL:
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# FROM TUTORIAL:
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        

def create_conv_layer(block, in_features, idx):
    # creates a convolutional layer including batch normalization and activation
    # input:
    # conv block dictionary, parsed from config file
    # in_features, number of output channels in previous layer
    # idx, # of the block within the entire network

    seq = nn.Sequential()
    
    ### conv layer ###
    # get all block params
    in_channels  = in_features
    out_channels = int(block['filters'])
    kernel_size  = int(block['size'])
    stride       = int(block['stride'])
    
    # FROM TUTORIAL:
    try:
        batch_norm = int(block['batch_normalize'])
        bias = False
    except:
        batch_norm = 0
        bias = True
    
    # FROM TUTORIAL
    if int(block['pad']):
        padding = (int(block['size'])-1) //2
    else:
        padding = 0
    
    seq.add_module('conv_{}'.format(idx),
                   nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    
    ### batch normalization layer ###
    if batch_norm:
        seq.add_module('batch norm_{}'.format(idx),
                       nn.BatchNorm2d(num_features = int(block['filters'])))
    
    ### activation layer ###
    if block['activation'] == 'leaky':
        seq.add_module('activation', nn.LeakyReLU())
    elif block['activation'] == 'linear':
        pass
    else:
        raise ValueError('Activation type note recognised/implemented: {}'.format(block['activation']))
                   
    return seq


def create_upsample_layer(block, idx):
    # creates an upsampling layer
    # input:
    # upsample block dictionary, parsed from config file
    # idx, # of the block within the entire network
    
    seq = nn.Sequential()
    seq.add_module('upsample_{}'.format(idx),
                   nn.Upsample(scale_factor = int(block['stride']),
                               mode         = 'bilinear'))
    
    return seq

def create_route_layer(block, idx):
    # creates a routing layer, concatenating two different feature maps together
    # input:
    # route block dictionary, parsed from config file
    # idx, # of the block within the entire network
    
    seq = nn.Sequential()
    seq.add_module('route_{}'.format(idx),
                   EmptyLayer())
    return seq

def create_shortcut_layer(block, idx):
    # creates a shortcut layer, adding a previous feature map to the current
    # input:
    # shortcut block dictionary, parsed from config file
    # idx, # of the block within the entire network
    
    seq = nn.Sequential()
    seq.add_module('shortcut_{}'.format(idx),
                   EmptyLayer())
    return seq

def create_detection_layer(block, idx):
    # creates a detection layer
    # input:
    # detection block dictionary, parsed from config file
    # idx, # of the block within the entire network

    mask    = [int(msk) for msk in block['mask'].split(',')]
    anchors = [int(anchor) for anchor in block['anchors'].split(',')]
    anchors = [[anchors[i], anchors[i+1]] for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]
#     classes :  80
#     num :  9
#     jitter :  .3
#     ignore_thresh :  .7
#     truth_thresh :  1
#     random :  1
    
    seq = nn.Sequential()
    seq.add_module('shortcut_{}'.format(idx),
                   DetectionLayer(anchors))
    return seq


def predict_transform(prediction, input_size, anchors, num_classes, gpu_enabled=False):
    """
    This function consists of two steps
    1: It takes a [batch_size, 3*(5+num_classes), h, w] feature map
       and transforms it to a list of single bbox predictions
    2: It takes the output of (1) and maps the bbox prediction parameters t_x, t_y, t_w, t_h, p_o
       to actual bboxes. For more information see the paper (yolo9000 or yolo v3)
    NB gpu_enabled is currently not properly implemented!
    input
    pred:        feature map
    input_size:  input size of the net's input image
    anchors:     list of anchors related to the current feature map (anchor=[x_prior, y_prior])
    num_classes: number of classes, 80 for COCO
    """
    pred      = prediction
    h, w      = pred.shape[2], pred.shape[3]
    C         = num_classes
    n_anchors = len(anchors)
    scale     = input_size // h * 1.
    
    # step 1
    assert (pred.shape[1] == n_anchors*(5+C)), \
        'predict_transform() expected 255 feature channel predictions. Received {}.'.format(pred.shape[1]) 

    pred = pred.permute(0, 2, 3, 1).contiguous()      # put the bboxes for single location in last dimension 
    pred = pred.view(-1, h*w*n_anchors, 5+C)          # squash all dims except for a single bbox dim (5+C)
    
    
    # step 2
    # this part is a bit trickier
    # the first 5 slices of pred (pred[:,:,:5])
    # correspond (in order) to these predictions: t_x, t_y, t_w, t_h, p_o
    
    ########## calculating bbox centers from t_x, t_y ##########
    # first create a meshed and warped version of the cell offset parameters c_x and c_y
    # N.B. we first need x then y (opposite of standard image dimension ordering of h then w)
    c_x, c_y    = np.meshgrid(np.arange(w), np.arange(h))               # create mesh coordinates
    c_x, c_y    = c_x.reshape(-1)[:, None], c_y.reshape(-1)[:, None]    # flatten mesh coordinates
    c_xy_offset = np.concatenate((c_x, c_y), axis=1)                    # concatenate x and y coordinates
    c_xy_offset = np.tile(c_xy_offset, (1,n_anchors)).reshape(-1,2)     # tile by the amount of anchors
    c_xy_offset = torch.tensor(c_xy_offset, dtype=torch.float)[None, :] # cast to torch, create batch dimensions
    
    torch.sigmoid_(pred[:, :, :2]).add_(c_xy_offset)                    # apply sigmoid, add cell offset
    pred[:, :, :2].mul_(scale)                                          # scale pred location to net's input size 
    
    ########## calculating bbox dimensions from t_w, t_h ##########
    anchors = np.tile(anchors, (h*w,1))                                 # tile by the amount for cells
    anchors = torch.tensor(anchors, dtype=torch.float)[None, :]         # cast to torch, create batch dimensions
    
    torch.exp_(pred[:, :, 2:4]).mul_(anchors)                           # apply exp and multiply by anchor dims
    
    
    ########## calculating objectness and class probabilities from p_o ##########
    pred[:, :, 4:] = torch.sigmoid(pred[:, :, 4:])         # sigmoid the objectness[4] & each class's[5:] score
        
    return pred



