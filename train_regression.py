from glinsun_dataset import GlinsunDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Model(nn.Module):
    '''
    TRAIN LOSS:
    VALID LOSS:
    '''

    def __init__(self):
        super(Model, self).__init__()
        self.extractor = torchvision.models.densenet121(True)
        self.embd = nn.Embedding(3, 2)

        self.head0 = nn.Sequential(nn.Linear(5, 1024), nn.ELU())
        self.regressor = nn.Sequential(nn.Linear(1024 + 1024, 2048), nn.ELU(), nn.Linear(2048, 4096), nn.ELU(), nn.Linear(4096, 19))

    def forward(self, img, gender, height, weight, age):
        # with torch.no_grad():

        features = self.extractor.features(img)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        gender = self.embd(gender)

        x = torch.cat((gender, height / 150., weight / 90., age / 60.), 1)

        x0 = self.head0(x)
        x1 = out
        x = torch.cat((x0, x1), 1)

        return self.regressor(x)


def train():
    torch.random.manual_seed(1)
    np.random.seed(1)

    model = Model().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True, min_lr=1e-4)

    # state_dict = torch.load('state_dict.torch')
    # model.load_state_dict(state_dict)

    dataset = GlinsunDataset("./Overall-Body-Database-20201017-corrected.csv",
                             "/home/sparky/body-dataset/")
    #
    # dataset = GlinsunDataset("./Overall-Body-Database-20201017-corrected.csv",
    #                          "/home/sparky/Documents/Projects/aibody-dataset/")

    train_set, val_set = torch.utils.data.random_split(dataset, [1600, 1892 - 1600])  # 2093, 1954

    train_loader = DataLoader(train_set,
                              batch_size=32,
                              num_workers=8,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=32,
                            num_workers=8,
                            shuffle=True)

    validation_record = np.inf

    training_history = []
    validation_history = []

    epoch = 0

    while True:

        model = model.train()

        total_loss = 0
        N = 0
        max_error = 0
        bigget_than = 0

        for idx, batch in enumerate(train_loader):

            gender_value, height_value, weight_value, age_value, target, img = batch
            target = target.cuda()

            out = model.forward(img.cuda(), gender_value.cuda(), height_value.unsqueeze(1).cuda(),
                                weight_value.unsqueeze(1).cuda(), age_value.unsqueeze(1).cuda())

            out = out[~target.isnan()]
            target = target[~target.isnan()]

            l1_loss = F.l1_loss(out, target, reduction='none')
            # loss = loss[~loss.isnan()]


            if max_error < l1_loss.max().item():
                max_error = l1_loss.max().item()

            total_loss += l1_loss.sum().item()
            N += l1_loss.shape[0]
            bigget_than += (l1_loss > 2).sum().item()

            # print(model.extractor.conv1.weight[:1])

            optim.zero_grad()
            l1_loss.mean().backward()
            optim.step()

        total_loss /= N

        print('TRAIN LOSS: ', total_loss, max_error, bigget_than)

        training_history.append(total_loss)
        # schedular.step(total_loss)

        with torch.no_grad():

            model = model.eval()

            total_loss = 0
            N = 0
            max_error = 0
            bigget_than = 0
            # go over validation multiple times (augmentation)
            for _ in range(1):

                for batch in val_loader:
                    gender_value, height_value, weight_value, age_value, target, img = batch

                    out = model.forward(img.cuda(), gender_value.cuda(), height_value.unsqueeze(1).cuda(),
                                        weight_value.unsqueeze(1).cuda(), age_value.unsqueeze(1).cuda())

                    out = out[~target.isnan()]
                    target = target[~target.isnan()]

                    loss = F.l1_loss(out, target.cuda(), reduction='none')

                    if max_error < loss.max().item():
                        max_error = loss.max().item()

                    total_loss += loss.sum().item()
                    N += loss.shape[0]
                    bigget_than += (loss > 2).sum().item()

            total_loss /= N
            print('VALID LOSS: ', total_loss, validation_record, max_error, bigget_than)

            validation_history.append(total_loss)

            print(out.detach().cpu().numpy()[:1], ' | ', target.detach().cpu().numpy()[:1])

            if validation_record > total_loss:
                validation_record = total_loss
                torch.save(model.state_dict(), 'state_dict.torch')
                print('saving ... new record ', validation_record)

            plt.figure()
            plt.plot(training_history)
            plt.savefig('training.png')
            plt.close()

            plt.figure()
            plt.plot(validation_history)
            plt.savefig('validation.png')
            plt.close()

            print('EPOCH ' + str(epoch) + ' is finished')
            epoch += 1


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    train()
