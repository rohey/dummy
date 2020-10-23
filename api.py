from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

from glinsun_dataset import GlinsunDataset
import gluoncv
import mxnet as mx
import os
from gluoncv.data.transforms.presets.segmentation import test_transform

ctx = mx.cpu()


def segmentation_mask(model, img):
    # using cpu
    img = mx.nd.array(np.asarray(img))
    img = test_transform(img, ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy() == 15.

    # from gluoncv.utils.viz import get_color_pallete
    # mask = get_color_pallete(predict, 'pascal_voc')

    # mask.save('output.png')
    # mmask = mpimg.imread('output.png')
    # plt.imshow(mask)
    # plt.show()

    return predict


class Model(nn.Module):
    '''
    TRAIN LOSS:
    VALID LOSS:
    '''


    def __init__(self):
        super(Model, self).__init__()
        self.extractor = torchvision.models.densenet121(False)
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

def extract_measurement(tensor):
    results = dict()
    results['MidNeckCircumference'] = tensor[0, 0].item()
    results['NeckCircumference1'] = tensor[0, 1].item()
    results['ShoulderWidth'] = tensor[0, 2].item()
    results['SleeveCircumference'] = tensor[0, 3].item()

    results['ArmLength1'] = tensor[0, 4].item()
    results['BicepsCircumference'] = tensor[0, 5].item()
    results['WristCircumference'] = tensor[0, 6].item()
    results['BreastCircumference'] = tensor[0, 7].item()

    results['UpperBreastCircumference'] = tensor[0, 8].item()
    results['LowerBreastCircumference'] = tensor[0, 9].item()
    results['WaistCircumference1'] = tensor[0, 10].item()
    results['AbodomenCircumference'] = tensor[0, 11].item()

    results['CoxaCircumference'] = tensor[0, 12].item()
    results['HipCircumference'] = tensor[0, 13].item()
    results['ThighCircumference'] = tensor[0, 14].item()
    results['CalfCircumference'] = tensor[0, 15].item()

    results['LegLength1'] = tensor[0, 16].item()
    results['FrontNeck2HipLength1'] = tensor[0, 17].item()
    results['BackNeck2HipLength1'] = tensor[0, 18].item()

    return results


class Predictor:

    def __init__(self, state_dict):
        self.model = Model().eval().cuda()
        self.model.load_state_dict(state_dict)

        self.bkgmodel = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True, ctx=ctx)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize
        ])

    def run(self, gender_value, height_value, weight_value, age_value, img):
        mask = segmentation_mask(self.bkgmodel, img)
        img = np.array(img)
        img[~mask] = 0
        img = Image.fromarray(img)
        img = self.transform(img)

        # plt.imshow(img.transpose(0, 2).transpose(0, 1).cpu().detach().numpy())
        # plt.show()

        img = img.unsqueeze(0)

        out = self.model.forward(img.cuda(), gender_value.cuda(), height_value.cuda(),
                                 weight_value.cuda(), age_value.cuda())

        reusults = extract_measurement(out)

        return reusults


# def test(predictor):
#     dataset = GlinsunDataset("./Overall-Body-Database-20201017-corrected.csv",
#                              "/home/sparky/body-dataset/")
#
#     loader = DataLoader(dataset,
#                         batch_size=1,
#                         num_workers=0,
#                         shuffle=False)
#
#
#     for idx, batch in enumerate(loader):
#         gender_value, height_value, weight_value, age_value, target, img = batch
#         target = target.cuda()
#
#         out = predictor.model.forward(img.cuda(), gender_value.cuda(), height_value.unsqueeze(1).cuda(),
#                             weight_value.unsqueeze(1).cuda(), age_value.unsqueeze(1).cuda())
#
#         errors = (target - out).abs()
#         errors = errors[~errors.isnan()].mean()
#
#
#         reusults = extract_measurement(out)
#
#         print(errors)


if __name__ == '__main__':
    state_dict = torch.load("state_dict.torch")
    predictor = Predictor(state_dict)
    # test(predictor)

    gender = torch.tensor([0])  # 0 - female; 1 - male; 2 - who knows
    height = torch.tensor([[153]])  # cm
    weight = torch.tensor([[49]])  # kg
    age = torch.tensor([[38]])  # seconds.. joke. years.
    img = Image.open('2081-张雪花.jpg')  # photo
    img = transforms.Resize(256)(img)

    reusults = predictor.run(gender, height, weight, age, img)
    print(reusults)
