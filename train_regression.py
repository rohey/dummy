from glinsun_dataset import GlinsunDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Model0(nn.Module):
    '''
    TRAIN LOSS:  37.867761850357056
    VALID LOSS:  4.957615613937378 4.8920416831970215
    '''

    def __init__(self):
        super(Model0, self).__init__()
        self.extractor = torchvision.models.resnet18(True)
        self.embd = nn.Embedding(3, 5)

        self.regressor = nn.Linear(520, 19)

    def forward(self, img, gender, height, weight, age):
        with torch.no_grad():
            x = self.extractor.conv1(img)
            x = self.extractor.bn1(x)
            x = self.extractor.relu(x)
            x = self.extractor.maxpool(x)

            x = self.extractor.layer1(x)
            x = self.extractor.layer2(x)
            x = self.extractor.layer3(x)
            x = self.extractor.layer4(x)
            x = self.extractor.avgpool(x)

        features = torch.flatten(x, 1)
        gender = self.embd(gender)

        x = torch.cat((features, gender, height, weight, age), 1)

        return self.regressor(x)


class Model1(nn.Module):

    def __init__(self):
        super(Model1, self).__init__()
        self.extractor = torchvision.models.resnet18(True)
        self.embd = nn.Embedding(3, 5)

        self.regressor = nn.Linear(520, 19)

    def forward(self, img, gender, height, weight, age):
        with torch.no_grad():
            x = self.extractor.conv1(img)
            x = self.extractor.bn1(x)
            x = self.extractor.relu(x)
            x = self.extractor.maxpool(x)

            x = self.extractor.layer1(x)
            x = self.extractor.layer2(x)
            x = self.extractor.layer3(x)

        x = self.extractor.layer4(x)
        x = self.extractor.avgpool(x)

        features = torch.flatten(x, 1)
        gender = self.embd(gender)

        x = torch.cat((features, gender, height, weight, age), 1)

        return self.regressor(x)


def train():
    torch.random.manual_seed(1)
    np.random.seed(1)

    model = Model1().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)

    dataset = GlinsunDataset("/home/sparky/body-dataset/Overall-Body-Database-20201009.csv",
                             "/home/sparky/body-dataset/")
    # dataset = GlinsunDataset("/home/sparky/Documents/Projects/aibody-dataset/Overall-Body-Database-20201009.csv",
    #                          "/home/sparky/Documents/Projects/aibody-dataset/")

    train_set, val_set = torch.utils.data.random_split(dataset, [1893, 200])

    train_loader = DataLoader(train_set,
                              batch_size=128,
                              num_workers=8,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=128,
                            num_workers=8,
                            shuffle=True)

    validation_record = np.inf

    training_history = []
    validation_history = []

    while True:

        model = model.train()

        total_loss = 0

        for batch in train_loader:
            gender_value, height_value, weight_value, age_value, target, img = batch

            out = model.forward(img.cuda(), gender_value.cuda(), height_value.unsqueeze(1).cuda(),
                                weight_value.unsqueeze(1).cuda(), age_value.unsqueeze(1).cuda())
            loss = F.l1_loss(out, target.cuda(), reduction='none')
            loss = loss[~loss.isnan()]
            loss = loss.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        print('TRAIN LOSS: ', total_loss)

        training_history.append(total_loss)
        schedular.step(total_loss)

        model = model.eval()

        total_loss = 0

        # go over validation multiple times (augmentation)
        for _ in range(10):

            for batch in val_loader:
                gender_value, height_value, weight_value, age_value, target, img = batch

                out = model.forward(img.cuda(), gender_value.cuda(), height_value.unsqueeze(1).cuda(),
                                    weight_value.unsqueeze(1).cuda(), age_value.unsqueeze(1).cuda())
                loss = F.l1_loss(out, target.cuda(), reduction='none')
                loss = loss[~loss.isnan()]
                loss = loss.mean()

                total_loss += loss.item()

        print('VALID LOSS: ', total_loss, validation_record)

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


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    train()
