#!/usr/bin/env python
# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
import math
from torchstat import stat
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchsummary
from dataProcess import dataloader
from Model1_10 import RLAttention as model

from utils import train_one_epoch, evaluate

if __name__ == '__main__':
    lrf = 0.01
    epochs =128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader, validation_dataloader = dataloader("RML2016.10a_dict.pkl")
    model = model(num_classes=11).to(device)

    #
    # weights_dict = torch.load("./weights/model-41.pth", map_location=device)
    # model.load_state_dict(weights_dict, strict=False)



    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=0.001)
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    total = sum([param.nelement() for param in model.parameters()])

    print(total)

    tb_writer = SummaryWriter()


    for epoch in range(0, epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=device,
                                                epoch=epoch)

        # scheduler.step()

        val_loss, val_acc,maxacc,max_ep = evaluate(model=model,
                                     data_loader=validation_dataloader,
                                     device=device,
                                     epoch=epoch,)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
