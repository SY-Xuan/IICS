from __future__ import absolute_import
from collections import OrderedDict

import torch
from ..utils import to_torch
import numpy as np


def extract_cnn_feature(model, inputs, norm=True):
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip
    model.eval()
    inputs = to_torch(inputs)

    n, c, h, w = inputs.size()

    ff = torch.FloatTensor(n, 2048).zero_().cuda()

    for i in range(2):
        if (i == 1):
            inputs = fliplr(inputs)
        inputs2 = inputs.cuda()
        if hasattr(model, "module"):
            outputs = model.module.backbone_forward(inputs2)
        else:
            outputs = model.backbone_forward(inputs2)
        outputs = outputs.view(outputs.size(0), outputs.size(1))

        ff += outputs * 0.5
    if norm:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

    return ff
