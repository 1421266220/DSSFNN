import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from loss import FocalLoss


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    all_snr = {}
    acc_snr = {}
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))

        _, predicted = torch.max(pred.data, 1)
        _, labelsed = torch.max(labels.data, 1)
        # loss = loss_function(pred, labels.to(device))
        acc = predicted == labelsed.to(device)
        accu_num += acc.sum()
        acc_numpy = acc.cpu().numpy()


        N = labels.size(0)
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -torch.sum(log_prob * labels.to(device)) / N
        #
        # criterion = FocalLoss(num_class=11)
        # loss = criterion(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                             accu_loss.item() / (
                                                                                                     step + 1),
                                                                                             accu_num.item() / sample_num,
                                                                                            )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    all_snr = {}
    acc_snr = {}
    for i in range(-20, 20, 2):
        acc_snr[i] = 0.1
        all_snr[i] = 0.1
    sample_num = 0
    confdata = {}
    for i in range(-20, 20, 2):
        confdata[i] = [[], []]
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, snrs = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_int = torch.tensor(pred, dtype=torch.int64)
        # labels = torch.tensor(pred,dtype=torch.int64)
        # accu_num += torch.eq(pred_int, labels.to(device)).sum()

        _, predicted = torch.max(pred.data, 1)
        _, labelsed = torch.max(labels.data, 1)
        # loss = loss_function(pred, labels.to(device))

        acc = predicted == labelsed.to(device)
        accu_num += acc.sum()
        acc_numpy = acc.cpu().numpy()
        snrs_numpy = snrs.cpu().numpy()
        for flag, pos in zip(acc_numpy, snrs_numpy):
            if flag == False:
                all_snr[pos] = all_snr[pos] + 1
            else:
                all_snr[pos] = all_snr[pos] + 1
                acc_snr[pos] = acc_snr[pos] + 1

        # 混淆矩阵数据
        predicted1 = predicted.cpu().numpy()
        labelsed1 = labelsed.cpu().numpy()

        for pre1, lab1, snr in zip(predicted1, labelsed1, snrs_numpy):
            confdata[snr][0].append(lab1)
            confdata[snr][1].append(pre1)
        N = labels.size(0)
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -torch.sum(log_prob * labels.to(device)) / N
        #
        # criterion = FocalLoss(num_class=11)
        # loss = criterion(pred, labels)
        accu_loss += loss.detach()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,)




    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
