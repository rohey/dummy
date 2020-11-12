from glinsun_dataset import GlinsunDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time


class Model(nn.Module):
    '''
    TRAIN LOSS:
    VALID LOSS:
    '''

    def __init__(self):
        super(Model, self).__init__()
        self.extractor = torchvision.models.densenet121(True)
        self.embd_gender = nn.Embedding(3, 2)
        self.embd_shape = nn.Embedding(9, 5)
        self.head0 = nn.Sequential(nn.Linear(5 + 5, 2048), nn.GELU())
        self.regressor = nn.Sequential(nn.Linear(1024 + 2048, 2048), nn.GELU(), nn.Dropout(0.25),
                                       nn.Linear(2048, 4096), nn.GELU(), nn.Dropout(0.5), nn.Linear(4096, 19))

    def forward(self, img, shape_value, gender, height, weight, age):
        # with torch.no_grad():

        features = self.extractor.features(img)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        gender = self.embd_gender(gender)
        shape = self.embd_shape(shape_value)

        x = torch.cat((gender, shape, height / 70., weight / 45., age / 30.), 1)

        x0 = self.head0(x)
        x1 = out
        x = torch.cat((x0, x1), 1)

        return self.regressor(x)


def train(index):
    torch.random.manual_seed(index + time.time())
    np.random.seed(index + int(time.time()))

    model = Model().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True, min_lr=1e-4)

    # state_dict = torch.load('state_dict.torch')
    # model.load_state_dict(state_dict)

    dataset = GlinsunDataset("./Overall-Body-Database-20201028.csv", "/home/sparky/body-dataset/")
    #
    # dataset = GlinsunDataset("./Overall-Body-Database-20201017-corrected.csv",
    #                          "/home/sparky/Documents/Projects/aibody-dataset/")

    train_set, val_set = torch.utils.data.random_split(dataset, [1600, 1846 - 1600])  # 2093, 1954

    train_loader = DataLoader(train_set,
                              batch_size=32,
                              num_workers=8,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=32,
                            num_workers=8,
                            shuffle=True)

    final_record = np.inf

    training_history = []
    validation_history = []

    epoch = 0
    no_progress = 0

    while True:

        model = model.train()

        total_train_loss = 0
        N = 0
        max_error = 0
        bigget_than = 0

        for idx, batch in enumerate(train_loader):

            shape_value, gender_value, height_value, weight_value, age_value, target, img = batch
            target = target.cuda()

            out = model.forward(img.cuda(), shape_value.cuda(), gender_value.cuda(), height_value.unsqueeze(1).cuda(),
                                weight_value.unsqueeze(1).cuda(), age_value.unsqueeze(1).cuda())

            out = out[~target.isnan()]
            target = target[~target.isnan()]

            l1_loss = F.l1_loss(out, target, reduction='none')
            # loss = loss[~loss.isnan()]

            if max_error < l1_loss.max().item():
                max_error = l1_loss.max().item()

            total_train_loss += l1_loss.sum().item()
            N += l1_loss.shape[0]
            bigget_than += (l1_loss > 2).sum().item()

            # print(model.extractor.conv1.weight[:1])

            optim.zero_grad()
            l1_loss.mean().backward()
            optim.step()

        total_train_loss /= N

        print('TRAIN LOSS: ', total_train_loss, max_error, bigget_than)

        training_history.append(total_train_loss)
        # schedular.step(total_loss)

        with torch.no_grad():

            model = model.eval()

            total_val_loss = 0
            N = 0
            max_error = 0
            bigget_than = 0
            # go over validation multiple times (augmentation)
            for _ in range(5):

                for batch in val_loader:
                    shape_value, gender_value, height_value, weight_value, age_value, target, img = batch

                    out = model.forward(img.cuda(), shape_value.cuda(), gender_value.cuda(), height_value.unsqueeze(1).cuda(),
                                        weight_value.unsqueeze(1).cuda(), age_value.unsqueeze(1).cuda())

                    out = out[~target.isnan()]
                    target = target[~target.isnan()]

                    loss = F.l1_loss(out, target.cuda(), reduction='none')

                    if max_error < loss.max().item():
                        max_error = loss.max().item()

                    total_val_loss += loss.sum().item()
                    N += loss.shape[0]
                    bigget_than += (loss > 2).sum().item()

            total_val_loss /= N
            print('VALID LOSS: ', total_val_loss, final_record, max_error, bigget_than)

            validation_history.append(total_val_loss)

            out = out.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            chosen = np.random.randint(0, target.shape[0])

            print(out[chosen], ' | ', target[chosen])

            final_loss = total_val_loss + total_train_loss * 0.1

            if final_record > final_loss:
                final_record = final_loss
                torch.save(model.state_dict(), 'state_dict_' + str(index) + '.torch')
                print('saving ... new record ', final_record)
                no_progress = 0
            else:
                no_progress += 1

            plt.figure()
            plt.yscale('log')
            plt.plot(training_history)
            plt.savefig('training' + str(index) + '.png')
            plt.close()

            plt.figure()
            plt.yscale('log')
            plt.plot(validation_history)
            plt.savefig('validation' + str(index) + '.png')
            plt.close()

            print('EPOCH ' + str(epoch) + ' is finished', no_progress, index)
            epoch += 1

            if no_progress > 100:
                print('finished training')
                return


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    for i in range(16):
        train(i)
