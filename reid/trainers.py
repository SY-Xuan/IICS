from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, LabelSmoothing
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, warm_up_epoch=-1):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.warm_up_epoch = warm_up_epoch

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))
            if epoch < self.warm_up_epoch:
                loss = loss * 0.1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'.format(
                          epoch, i + 1, len(data_loader), batch_time.val,
                          batch_time.avg, data_time.val, data_time.avg,
                          losses.val, losses.avg, precisions.val,
                          precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        elif isinstance(self.criterion, LabelSmoothing):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class IntraCameraTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        pass

    def _forward(self, inputs, targets, i):
        outputs = self.model(inputs, i)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        elif isinstance(self.criterion, LabelSmoothing):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec

    def train(self,
              cluster_epoch,
              epoch,
              data_loader,
              optimizer,
              print_freq=1,
              ):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        data_loader_size = min([len(l) for l in data_loader])
        for i, inputs in enumerate(zip(*data_loader)):
            data_time.update(time.time() - end)
            for domain, domain_input in enumerate(inputs):
                imgs, _, pids, _ = domain_input
                imgs = imgs.cuda()
                targets = pids.cuda()

                loss, prec1 = self._forward(imgs, targets, domain)
                if domain == 0:
                    loss_sum = loss
                else:
                    loss_sum += loss
                losses.update(loss.item(), targets.size(0))
                precisions.update(prec1, targets.size(0))

            if cluster_epoch < self.warm_up_epoch:
                loss_sum = loss_sum * 0.1

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Cluster_Epoch: [{}]\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'.format(
                          cluster_epoch, epoch, i + 1, data_loader_size,
                          batch_time.val, batch_time.avg, data_time.val,
                          data_time.avg, losses.val, losses.avg,
                          precisions.val, precisions.avg))


class InterCameraTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 entropy_criterion,
                 triple_criterion,
                 warm_up_epoch=-1,
                 multi_task_weight=1.):
        super(InterCameraTrainer, self).__init__(model, entropy_criterion,
                                                 warm_up_epoch)
        self.triple_critetion = triple_criterion
        self.multi_task_weight = multi_task_weight

    def _parse_data(self, inputs):
        pass

    def _forward(self, inputs, targets):
        prob, distance = self.model(inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss_entropy = self.criterion(prob, targets)
            prec_entropy, = accuracy(prob.data, targets.data)
            prec_entropy = prec_entropy[0]
        elif isinstance(self.criterion, LabelSmoothing):
            loss_entropy = self.criterion(prob, targets)
            prec_entropy, = accuracy(prob.data, targets.data)
            prec_entropy = prec_entropy[0]

        loss_triple, prec_triple = self.triple_critetion(distance, targets)

        return loss_entropy, prec_entropy, loss_triple, prec_triple

    def train(self,
              cluster_epoch,
              epoch,
              data_loader,
              optimizer,
              print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_entropy = AverageMeter()
        precisions_entropy = AverageMeter()
        losses_triple = AverageMeter()
        precisions_triple = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs, _, pids, _ = inputs
            imgs = imgs.cuda()
            targets = pids.cuda()

            loss_entropy, prec_entropy, loss_triple, prec_triple = self._forward(imgs, targets)

            loss = loss_triple * self.multi_task_weight + loss_entropy

            losses_entropy.update(loss_entropy.item(), targets.size(0))
            precisions_entropy.update(prec_entropy, targets.size(0))
            losses_triple.update(loss_triple.item(), targets.size(0))
            precisions_triple.update(prec_triple, targets.size(0))

            if cluster_epoch < self.warm_up_epoch:
                loss = loss * 0.1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Cluster_Epoch: [{}]\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_Entropy {:.3f} ({:.3f})\t'
                      'Prec_Entropy {:.2%} ({:.2%})\t'
                      'Loss_Triple {:.3f} ({:.3f})\t'
                      'Prec_Triple {:.2%} ({:.2%})\t'.format(
                          cluster_epoch, epoch, i + 1, len(data_loader),
                          batch_time.val, batch_time.avg, data_time.val,
                          data_time.avg, losses_entropy.val, losses_entropy.avg,
                          precisions_entropy.val, precisions_entropy.avg,
                          losses_triple.val, losses_triple.avg,
                          precisions_triple.val, precisions_triple.avg,
                          ))