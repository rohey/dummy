from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from PIL import ImageOps
from glinsun_dataset import segmentation_mask, convert_shape
import gluoncv
import mxnet as mx
import os
from gluoncv.data.transforms.presets.segmentation import test_transform
from efficientnet_pytorch import EfficientNet
ctx = mx.cpu()


class SimpleModel(nn.Module):
    '''
    https://www.kaggle.com/artkulak/inference-45-55-600-epochs-tuned-effnet-b5-30-ep
    https://github.com/lukemelas/EfficientNet-PyTorch
    https://github.com/artkulak/osic-pulmonary-fibrosis-progression

    TRAIN LOSS:
    VALID LOSS:
    '''

    def __init__(self, index):
        super(SimpleModel, self).__init__()
        self.embd_gender = nn.Embedding(3, 2)
        self.embd_shape = nn.Embedding(9, 5)
        self.regressor = nn.Sequential(nn.Linear(10, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 19))

    def forward(self, img, shape_value, gender, height, weight, age):
        # with torch.no_grad():
        gender = self.embd_gender(gender)
        shape = self.embd_shape(shape_value)
        x = torch.cat((gender, shape, height / 70., weight / 45., age / 30.), 1)
        return self.regressor(x)




class EfficientModel(nn.Module):
    '''
    https://www.kaggle.com/artkulak/inference-45-55-600-epochs-tuned-effnet-b5-30-ep
    https://github.com/lukemelas/EfficientNet-PyTorch
    https://github.com/artkulak/osic-pulmonary-fibrosis-progression

    TRAIN LOSS:
    VALID LOSS:
    '''

    def __init__(self, index):
        super(EfficientModel, self).__init__()
        override_params = dict()
        override_params['include_top'] = False

        index = index % 4

        self.extractor = EfficientNet.from_name('efficientnet-b' + str(index), include_top=False)

        self.embd_gender = nn.Embedding(3, 2)
        self.embd_shape = nn.Embedding(9, 5)


        if index == 0 or index == 1:
            fnum = 1280
        elif index == 2:
            fnum = 1408
        elif index == 3:
            fnum = 1536

        self.regressor = nn.Sequential(nn.Linear(fnum + 10, 2048), nn.Dropout(0.5), nn.Linear(2048, 19))

    def forward(self, img, shape_value, gender, height, weight, age):
        # with torch.no_grad():

        features = self.extractor.forward(img)
        x0 = torch.flatten(features, 1)

        gender = self.embd_gender(gender)
        shape = self.embd_shape(shape_value)

        x1 = torch.cat((gender, shape, height / 70., weight / 45., age / 30.), 1)

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

    def __init__(self):
        folder = '12_11_2020'

        self.simple_dicts = []
        for i in range(3):
            state_dict = torch.load(folder + "/simple_net_" + str(i) + ".torch")[0]
            self.simple_dicts.append(state_dict)

        self.efficient_dicts = []
        for i in range(3):
            state_dict = torch.load(folder + "/efficient_net_" + str(i) + ".torch")[0]
            self.efficient_dicts.append(state_dict)



        self.seg_model = gluoncv.model_zoo.get_model('deeplab_resnet152_voc', pretrained=True, ctx=ctx)
        self.box_model = gluoncv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


    def run(self, id, gender_value, shape_type, height_value, weight_value, age_value, pilimg):

        img = transforms.Resize(512)(pilimg)
        img = segmentation_mask(self.seg_model, self.box_model, img)
        img = Image.fromarray(img)

        desired_size = max(img.height, img.width)
        delta_w = desired_size - img.width
        delta_h = desired_size - img.height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_im = ImageOps.expand(img, padding)









        simple_predictions = []
        for i in range(3):
            model = SimpleModel(i).eval().cuda()
            model.load_state_dict(self.simple_dicts [i])
            out = model.forward(None, shape_type.cuda(), gender_value.cuda(), height_value.cuda(),
                                      weight_value.cuda(), age_value.cuda())
            simple_predictions.append(out)
        simple_predictions = torch.cat(simple_predictions, 0).mean(0, keepdim=True)



        efficient_predictions = []
        for i in range(3):
            model = EfficientModel(i).eval().cuda()
            model.load_state_dict(self.efficient_dicts [i])

            image_size = model.extractor._global_params.image_size

            # add scale
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize
            ])

            img = transform(new_im)
            img = img.unsqueeze(0)

            out = model.forward(img.cuda(), shape_type.cuda(), gender_value.cuda(), height_value.cuda(),
                                      weight_value.cuda(), age_value.cuda())
            efficient_predictions.append(out)
        efficient_predictions = torch.cat(efficient_predictions, 0).mean(0, keepdim=True)


        if id == 0:
            out = simple_predictions
        elif id == 1:
            out = efficient_predictions
        else:
            out = simple_predictions * 0.75 + efficient_predictions * 0.25

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
    with torch.no_grad():
        predictor = Predictor()
        # test(predictor)
        index = 2 # 0 - variant #1, 1 - variant #2,  2 - the combination of #1 and #2
        gender = torch.tensor([1])  # 0 - female; 1 - male; 2 - who knows
        height = torch.tensor([[187]])  # cm
        weight = torch.tensor([[120]])  # kg
        age = torch.tensor([[38]])  # seconds.. joke. years.


        """
        def convert_shape(gender, shape_type):
            if (gender == "male") or (gender == "Male"):
                if shape_type == "big belly":
                    return 0
                elif shape_type == "small belly":
                    return 1
                elif shape_type == "standard":
                    return 2
                elif shape_type == "oval":
                    return 3
                elif shape_type == "trapezoid":
                    return 4
                else:
                    raise Exception()
            else:
                if shape_type == "rectangle":
                    return 5
                if shape_type == "triangle":
                    return 6
                if shape_type == "hourglass":
                    return 7
                if shape_type == "inverted triangle":
                    return 8
                else:
                    raise Exception()
        """
        shape_type = torch.tensor([8])


        img = Image.open('/home/sparky/Downloads/20201019-Pics/16_1.jpg')  # photo

        reusults = predictor.run(index, gender, shape_type, height, weight, age, img)
        print(reusults)
