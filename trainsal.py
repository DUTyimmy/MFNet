import os
import time
import torch
import numpy as np
import torch.nn as nn

from datetime import datetime
from utils.imsave import imsave
from utils.pamr import BinaryPamr
from utils.datainit import valdatainit
from utils.evaluateFM import get_FM
from tensorboardX import SummaryWriter

# Enter [tensorboard --logdir=log] in [Terminal in Pycharm IDE] or [terminal in Ubuntu OS] or [Win+r-cmd in Windows OS]
# to use tensorboard to visualize the training process. You can also use [import torch.utils.tensorboard].

writer = SummaryWriter('log')
loss_1, loss_2, loss_self, loss_3 = 0.0, 0.0, 0.0, 0.0


class TrainSal(object):

    def __init__(self, model, optimizer_model, train_loader, val_loader, outpath, max_epoch=20, stage=2):
        self.model = model
        self.optim_model = optimizer_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epoch = int(max_epoch)
        self.stage = int(stage)
        self.outpath = outpath

        self.BCEloss = nn.BCELoss()
        self.sshow = 20
        self.iteration = 0
        self.epoch = 0

    @staticmethod
    def rgb2grey(images):
        mean = np.array([0.447, 0.407, 0.386])
        std = np.array([0.244, 0.250, 0.253])

        images1, images2, images3 = images[:, 0:1, :, :], images[:, 1:2, :, :], images[:, 2:3, :, :]
        images1 = images1 * std[0] + mean[0]
        images2 = images2 * std[1] + mean[1]
        images3 = images3 * std[2] + mean[2]
        img_grey = images1 * 0.299 + images2 * 0.587 + images3 * 0.114
        return img_grey

    @staticmethod
    def l2loss(sal1, sal2):
        mse = (sal1 - sal2).norm(2).pow(2)
        return mse / (sal1.shape[2] * sal2.shape[3])

    def mutualloss(self, sal1, sal2):
        # mseloss can also be adopted, which encourages similar performance
        sal1_self, sal2_self = sal1.clone().detach(), sal2.clone().detach()
        loss = self.BCEloss(sal1, sal2_self) + self.BCEloss(sal2, sal1_self)
        return loss/2

    @staticmethod
    def run_ctr(sal):
        pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        boundary = torch.abs(pool(sal) - (1 - pool(1 - sal)))
        return boundary

    @staticmethod
    def run_pamr(img, sal):
        lbl_self = BinaryPamr(img, sal.clone().detach(), binary=0.4)
        return lbl_self

    def train_epoch(self):
        for batch_idx, (img, lbl1, lbl2) in enumerate(self.train_loader):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            if self.epoch >= self.max_epoch * len(self.train_loader):
                break
            img, lbl1, lbl2 = img.cuda(), lbl1.cuda().unsqueeze(1), lbl2.cuda().unsqueeze(1)

            sal1, sal2, sal3 = self.model.forward(img)

            # BCEloss for the 1st DF
            loss1 = self.BCEloss(sal1, lbl1)

            # BCEloss for the 2nd DF
            loss2 = self.BCEloss(sal2, lbl2)

            # The self-supervision term between 1st DF and 2nd DF
            loss12 = self.mutualloss(sal1, sal2)

            # Guidance loss for the final saliency decoder
            lbl_tea = self.run_pamr(img, (sal1+sal2)/2)
            loss3 = self.BCEloss(sal3, lbl_tea)

            loss = loss1 + loss2 + loss3 + 2*loss12

            self.optim_model.zero_grad()
            loss.backward()
            self.optim_model.step()

            global loss_1, loss_2, loss_self, loss_3
            loss_1 += loss1.item()
            loss_2 += loss2.item()
            loss_self += loss12.item()
            loss_3 += loss3.item()

            # ------------------------------ information exhibition and visualization --------------------------- #
            if iteration % self.sshow == (self.sshow - 1):
                print('|| Time: %s,\t\tStage: %1d,\t\t\tEpoch: %2d/%2d,\t\tIter: %2d/%2d,\t\t||\n'
                      '|| Loss1: %.4f,\t\tLoss2: %.4f,\t\tLoss3: %.4f,\t\tLoss_self: %.4f\t||\n' %
                      (str(datetime.now().replace(microsecond=0))[11:],
                       self.stage, self.epoch + 1, self.max_epoch, batch_idx + 1, len(self.train_loader),
                       loss_1 / self.sshow, loss_2 / self.sshow, loss_3 / self.sshow, loss_self / self.sshow))

                # tensorboard-scale
                writer.add_scalar('loss of decoder1', loss_1 / self.sshow, iteration + 1)
                writer.add_scalar('loss of decoder2', loss_2 / self.sshow, iteration + 1)
                writer.add_scalar('loss of decoder3', loss_3 / self.sshow, iteration + 1)
                writer.add_scalar('loss of self-supervision', loss_self / self.sshow, iteration + 1)
                # tensorboard-image
                img_grey = self.rgb2grey(img.clone()[0].unsqueeze(0))
                image = torch.cat((img_grey[0], lbl1[0], lbl2[0], sal1[0],
                                   sal2[0], lbl_tea[0], sal3[0]), 0)
                image = torch.unsqueeze(image, 0).transpose(0, 1)
                writer.add_images('sal maps', image, iteration + 1, dataformats='NCHW')

                loss_1, loss_2, loss_self, loss_3 = 0.0, 0.0, 0.0, 0.0

    def train(self):
        best_mae, best_f = 0.0, 0.0

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.model.train()
            self.train_epoch()
            print('')

            if self.val_loader is not None:
                # ---------------- validation ---------------- #
                print('\nValidating .....   \n[  ', end='')
                valdatainit()
                self.model.eval()
                total_num = len(self.val_loader)
                count_num = int(total_num / 10)

                start_time = time.time()
                for idx, (data, name, size) in enumerate(self.val_loader):
                    _, _, sal = self.model(data.cuda())

                    sal = sal.squeeze().cpu().detach()

                    imsave(os.path.join('data/val_map', name[0] + '.png'), sal, size, factor=False)

                    if idx % count_num == count_num - 1:
                        print((str(round(int(idx + 1) / total_num * 100))) + '.0 %  ', end='')

                print('],  finished,  ', end='')
                final_time = time.time()
                print('cost %d seconds. ' % (final_time - start_time))

                # ---------------- evaluation ---------------- #
                print("\nEvaluating .....")
                f, mae = get_FM(salpath='data/val_map/', gtpath='data/ECSSD/mask/')
                if f > best_f:
                    best_mae, best_f = mae, f
                    savename = ('%s/sal_stage_%d.pth' % (self.outpath, self.stage))
                    torch.save(self.model.state_dict(), savename)

                print('this F_measure:% .4f' % f, end='\t\t')
                print('this MAE:% .4f' % mae)
                print('best F_measure:% .4f' % best_f, end='\t\t')
                print('best MAE:% .4f' % best_mae, end='\n\n')

                writer.add_scalar('F-measure', f, epoch+1)
                writer.add_scalar('MAE', mae, epoch+1)

            else:
                savename = ('%s/sal_stage_%d.pth' % (self.outpath, self.stage))
                torch.save(self.model.state_dict(), savename)

            if self.epoch >= self.max_epoch * len(self.train_loader):
                break
