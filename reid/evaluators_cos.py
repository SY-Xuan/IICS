from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter

import numpy as np

def fresh_bn(model, data_loader):
    model.train()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs)
        print('Fresh BN: [{}/{}]\t'.format(i, len(data_loader)))

        
def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None, use_cpu=False):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = 1 - torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    if use_cpu:
        dist = 1 - np.matmul(x.cpu().numpy(), y.cpu().numpy().T)
        dist = np.array(dist)
    else:
        dist = 1 - torch.mm(x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), return_mAP=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        # 'allshots': dict(separate_camera_set=False,
        #                  single_gallery_shot=False,
        #                  first_match_break=False),
        # 'cuhk03': dict(separate_camera_set=True,
        #                single_gallery_shot=True,
        #                first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}'
          .format('market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, 
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    if return_mAP:
        return cmc_scores['market1501'][0], mAP
    else:
        return cmc_scores['market1501'][0]


class Evaluator(object):
    def __init__(self, model, use_cpu=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.use_cpu = use_cpu

    def evaluate(self, data_loader, query, gallery, metric=None, return_mAP=False):
        features, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric, use_cpu=self.use_cpu)
        return evaluate_all(distmat, query=query, gallery=gallery, return_mAP=return_mAP)

    def fresh_bn(self, data_loader):
        fresh_bn(self.model, data_loader)
