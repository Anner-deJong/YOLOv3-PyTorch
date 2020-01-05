# This file contains helper functions and layers for the Yolo v3 network

import cv2
import torch
import torch.nn as nn
import numpy as np


#####################################
### pytorch model layer functions ###
#####################################


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


######################################################
### model output into prediction warping functions ###
######################################################

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


##################################################
### Output/prediction/detection warp functions ###
##################################################

def pred2corners(pred):
    # prediction should contain 3 dim, with the last one consisting of: x, y, w, h
    # these are the centers and the dimensions of a prediction 
    # this get's mapped to left top corner x, y and right bottom corner x, y
    
    corners = torch.zeros_like(pred, dtype=pred.dtype)
    corners[:, :, 0] = pred[:, :, 0] - pred[:, :, 2]/2.
    corners[:, :, 1] = pred[:, :, 1] - pred[:, :, 3]/2.
    corners[:, :, 2] = pred[:, :, 0] + pred[:, :, 2]/2.
    corners[:, :, 3] = pred[:, :, 1] + pred[:, :, 3]/2.

    return corners

def IoU(corners1, corners2):
    # calculates the IoU between two 1D length 4 arrays consisting of:
    # left, top, right, bottem coordinates respectively
    x1_l, y1_t, x1_r, y1_b = corners1
    x2_l, y2_t, x2_r, y2_b = corners2
    
    # no horizontal overlap
    if (x1_r <= x2_l) or (x1_l >= x2_r):
        return 0
    
    # no vertical overlap
    if (y1_b <= y2_t) or (y1_t >= y2_b):
        return 0
    
    # else: some overlap exists
    # overlap (ovl) border values
    ovl_l = torch.max(x1_l, x2_l)
    ovl_r = torch.min(x1_r, x2_r)
    ovl_t = torch.max(y1_t, y2_t)
    ovl_b = torch.min(y1_b, y2_b)
    
    # surfaces
    A_1   = (x1_r - x1_l) * (y1_b - y1_t)
    A_2   = (x2_r - x2_l) * (y2_b - y2_t)
    A_ovl = (ovl_r - ovl_l) * (ovl_b - ovl_t)

    return (A_ovl / (A_1 + A_2 - A_ovl))


##########################################
### helper function to get class names ###
##########################################

def load_classes(class_file='./cfg_weights_utils/coco.names'):
    # reads in a text file with line separated class names, in order
    # returns a dictionary with class_name -> ind, and reverse, mapping
    classes = {}
    
    with open(class_file) as f:
        class_names = [name.rstrip('\n') for name in f]
        for i, class_name in enumerate(class_names):
            classes[i]          = class_name
            classes[class_name] = i
    
    classes['n_classes'] = i+1
    
    return classes


#########################################################
### image pipeline (preprocessing, drawing) functions ###
#########################################################

class ImgUtils():

    def preprocess(img, resolution=416):
        # assuming an image freshly loaded via opencv: numpy & BGR
        img    = img[:, :, ::-1]                  # inverse color channels

        h, w   = img.shape[:2]
        scale  = resolution / max(h, w)
        h, w   = int(h*scale), int(w*scale)
        img    = cv2.resize(img, (w, h))          # resize while keeping ratio

        img    = img.astype(np.float32) / 255.    # cast as float and normalize

        # add padding and keep track of the padding offset (-> [x, y, scale])
        if (h<w):
            pad_up    = np.ones( shape=((w-h) // 2, w, 3),          dtype=np.float32) * 128./255.
            offset    = [0, pad_up.shape[0], scale]
            pad_down  = np.ones( shape=(w-h-offset[1], w, 3), dtype=np.float32) * 128./255.
            img       = np.concatenate( (pad_up, img, pad_down), axis=0)
        else:
            pad_left  = np.ones( shape=( h, (h-w) // 2, 3),            dtype=np.float32) * 128./255.
            offset    = [pad_left.shape[1], 0, scale]
            pad_right = np.ones( shape=( h, h-w-offset[0], 3), dtype=np.float32) * 128./255.
            img       = np.concatenate( (pad_left, img, pad_right), axis=1)

        img = img.transpose(2, 0, 1)[None]       # HWC -> BCHW
        img = torch.tensor(img)

        return img, offset

    def draw_box(img, offset, cls_name, detection):
        # offset contains [x, y, scale] offsets
        # detection is a torch vector with:
        # [x_left, y_up, x_down, y_right, objectness score, cls score, cls index]

        # first offset and scale the detection coordinates
        x_l = int((detection[0].item() - offset[0]) / offset[2])
        y_u = int((detection[1].item() - offset[1]) / offset[2])
        x_r = int((detection[2].item() - offset[0]) / offset[2])
        y_b = int((detection[3].item() - offset[1]) / offset[2])

        # detection bounding box
        cv2.rectangle(img, (x_l, y_u), (x_r, y_b), (0, 0, 255), 5)
        
        # detection text box on top
        box_width = len(cls_name) * 20
        cv2.rectangle(img, (x_l-3, y_u-35), (x_l+box_width, y_u), (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, cls_name, (x_l+5, y_u-10), font, fontScale=.9,
                    color=(255,255,255), thickness=2)
        

        return img

    def annotate_img(img, offset, det_dict):
        ann_img = img.copy()

        for cls, dets in det_dict.items():
            for i in range(dets.shape[0]): # first dim represent all separate instances for a class    
                ann_img = ImgUtils.draw_box(ann_img, offset, cls, dets[i])

        return ann_img
    