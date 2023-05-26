#!/usr/bin/env python
# coding=utf-8
import pickle
import random
from numpy import linalg as la
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from dataGet import GetLoader, GetLoader_train
from getAPdata import getAPdata
from myModel.getspdata import getSPdata


def dataloader(fileLocation, batch_size=400, shuffle=True, val_rate=0.2):
    with open(fileLocation, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], p.keys())))), [1, 0])
    X = []
    label = []
    SNRs = []
    for mod in mods:
        for snr in snrs:
            X.append(p[(mod, snr)])
            for i in range(p[(mod, snr)].shape[0]):
                label.append(mod)
                SNRs.append(snr)

    X = np.vstack(X)


    encoder = LabelEncoder()
    encoder.fit(label)
    lu = np.unique(label)
    print(lu)


    label_num = encoder.transform(label)

    # one-hot编码


    xTrain, xValidation, yTrain, yValidation, snrTrain, snrValidation = train_test_split(X,
                                                                                         label_num, SNRs,
                                                                                         test_size=0.15,random_state=2016,shuffle=True)


    # 测试集one-hot
    label_unique_yValidation = np.unique(yValidation)
    yValidation = torch.nn.functional.one_hot(torch.tensor(yValidation).to(torch.int64), len(label_unique_yValidation))





    xTrain, yTrain = Rotate_DA(xTrain.transpose((0, 2, 1)), yTrain)
    xTrain = xTrain.transpose((0, 2, 1))

    # xTrain, yTrain = Gaussian_DA(xTrain.transpose((0, 2, 1)), yTrain)

    # xTrain = xTrain.transpose((0, 2, 1))

    label_unique_yTrain = np.unique(yTrain)
    yTrain = torch.nn.functional.one_hot(torch.tensor(yTrain).to(torch.int64), len(label_unique_yTrain))

    xTrain1 = amplitudeToPhase(xTrain, 128)
    xTrain1 = normalizeData(xTrain1, 128)
    xTrain1 = xTrain1.transpose(0, 2, 1)

    xValidation1 = amplitudeToPhase(xValidation, 128)
    xValidation1 = normalizeData(xValidation1, 128)
    xValidation1 = xValidation1.transpose(0, 2, 1)

    xTrain = xTrain.transpose(0, 2, 1)
    xTrain = normalizeData(xTrain)
    xTrain = xTrain.transpose(0, 2, 1)

    xValidation = xValidation.transpose(0, 2, 1)
    xValidation = normalizeData(xValidation)
    xValidation = xValidation.transpose(0, 2, 1)

    xTrain1 = torch.tensor(xTrain1, dtype=torch.float32)
    xTrain = torch.tensor(xTrain, dtype=torch.float32)

    xValidation1 = torch.tensor(xValidation1, dtype=torch.float32)
    xValidation = torch.tensor(xValidation, dtype=torch.float32)
    # X3 = torch.tensor(X3, dtype=torch.float32)

    xTrain = torch.cat((xTrain1, xTrain), dim=1)
    xValidation = torch.cat((xValidation1, xValidation), dim=1)





    # creating training, testing and validation set
    # training set contains 70% of the total data, validation and test set contains 15% each
    # SNRs are also split so as to check the classification accuracy of each SNR
    train_dataset = GetLoader_train(xTrain, yTrain)
    validation_dataset = GetLoader(xValidation, yValidation, snrValidation)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=2)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                                       num_workers=2)
    return train_dataloader, validation_dataloader


def normalizeData(x, length=128):
    print('Normalizing: ', x.shape)
    for i in range(x.shape[0]):
        x[i, :, 0] = x[i, :, 0] / np.linalg.norm(x[i, :, 0], 2)
    return x


# function to change amplitude to phase
def amplitudeToPhase(x, length=128):
    xComplex = x[:, 0, :] + 1j * x[:, 1, :]
    xAmplitude = np.abs(xComplex)
    xAngle = np.arctan2(x[:, 1, :], x[:, 0, :]) / np.pi
    xAmplitude = np.reshape(xAmplitude, (-1, 1, length))
    xAngle = np.reshape(xAngle, (-1, 1, length))
    x = np.concatenate((xAmplitude, xAngle), axis=1)
    x = np.transpose(np.array(x), (0, 2, 1))
    return x


def normalize(X):
    X1 = X[:, 0:2, :]
    X2 = X[:, 2:4, :]
    X3 = X[:, 4:6, :]

    X1 = np.transpose(X1, (0, 2, 1))
    X1 = normalizeData(X1, 128)
    X1 = np.transpose(X1, (0, 2, 1))

    return torch.cat((X1, X2, X3), dim=1)


def easear(X, label, snrs):
    X_ = []
    label_ = []
    SNRs_ = []
    for x, l, s in zip(X, label, snrs):
        p0 = random.uniform(0, 1)
        if p0 < 0.3:
            max = x.max()
            min = x.min()
            flag = True
            while flag:
                w_r = random.randint(0, 127)
                h_r = random.randint(0, 1)
                re = random.randint(0, 25)
                if re + w_r <= 127:
                    for i in range(w_r, re + w_r):
                        tmp = random.uniform(min, max)
                        x[0, i] = tmp
                        x[1, i] = tmp
                    flag = False
                X_.append(x)
                label_.append(l)
                SNRs_.append(s)
    return X_, label_, SNRs_



def hunluan(X, label):
    X_ = []
    label_ = []
    for x, l in zip(X, label):
        p0 = random.uniform(0, 1)
        if p0 < 1:
            flag = True
            while flag:
                w_r = random.randint(0, 127)
                re = random.randint(0, 25)
                if re + w_r <= 127 and re != 0:
                    # 切片

                    cat = x[:, w_r:re + w_r]
                    x = np.delete(x,range(w_r,re + w_r),axis=1)

                    pos = random.randint(0, len(x)-1)
                    x1 = x[:,:pos]
                    x2 = x[:,pos:]

                    x = np.append(x1,cat,axis=1)
                    x = np.append(x, x2, axis=1)
                    flag = False
            X_.append(x)
            label_.append(l)

    return np.array(X_), label_
# 旋转
def rotate_matrix(theta):
    m = np.zeros((2,2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    print(m)
    return m

def Rotate_DA(x, y):
    [N, L, C] = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3*np.pi/2))

    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T


    return x_DA, y_DA

def Shift_DA(x, y):
    [N, L, C] = np.shape(x)

    x_shift1 = np.zeros((N, L, C))
    x_shift1[:, 0 : int(3*L/4),:]= x[:, int(L/4):L, :]
    x_shift1[:, int(3*L/4) : L,:]= x[:, 0 : int(L/4), :]

    x_shift2 = np.zeros((N, L, C))
    x_shift2[:, 0: int(2*L/4),:]= x[:, int(2*L/4) : L, :]
    x_shift2[:, int(2*L/4): L,:]= x[:, 0 : int(2*L/4), :]

    x_shift3 = np.zeros((N, L, C))
    x_shift3[:, 0: int(1*L/4),:]= x[:, int(3*L/4) : L, :]
    x_shift3[:, int(1*L/4): L,:]= x[:, 0 : int(3*L/4), :]

    x_DA = np.vstack((x, x_shift1, x_shift2,x_shift3))

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA


def Gaussian_DA(x, y):
    [N, L, C] = np.shape(x)

    gaussian_noise1 = np.zeros([N, L, C])
    for n in range(N):
        gaussian_noise1[n,:,:] = np.random.normal(0, 0.0005, size=(1, L, C))

    gaussian_noise2 = np.zeros([N, L, C])
    for n in range(N):
        gaussian_noise2[n,:,:] = np.random.normal(0, 0.001, size=(1, L, C))

    gaussian_noise3 = np.zeros([N, L, C])
    for n in range(N):
        gaussian_noise3[n,:,:] = np.random.normal(0, 0.002, size=(1, L, C))


    x_DA = np.vstack((x, x+gaussian_noise1, x+gaussian_noise2, x+gaussian_noise3))

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA

if __name__ == '__main__':
    dataloader("RML2016.10a_dict.pkl")
