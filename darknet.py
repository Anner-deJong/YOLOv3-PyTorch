# this file will contain the underlying network of Yolo v3

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def parse_cfg(cfgfile):
    """
    Parse a config file that states the network architecture.
    Official Yolo v3 config can be found online and in ./cfg/yolov3.cfg
    
    Returns a list of dicts, each dict describing a layer in the network
    """
    blocks = []

    with open(cfgfile) as f:
        for line in file         
