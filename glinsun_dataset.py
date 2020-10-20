import torch
import torch.utils.data as data
import mxnet

import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import gluoncv
import mxnet as mx
import os
from gluoncv.data.transforms.presets.segmentation import test_transform

# 588,female,50,149.5,5740%,rectangle,2748-0671_1.png,523-0671_1.png,NaN,NaN,NaN,NaN,NaN,NaN,34,38,40,49,47,30,17,94,92.5,93,87,88,93,95,56,34.5,90,59,55,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN

ctx = mx.cpu()

def segmentation_mask(model, img):
    # using cpu
    img = mxnet.nd.array(np.asarray(img))
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

class GlinsunDataset(data.Dataset):

    def __init__(self, csv_file, img_path):
        csv = pd.read_csv(csv_file, encoding="GBK")
        # csv = csv.replace(np.nan, '', regex=True)
        # csv.query('bkgFreePose != "NaN"', inplace=True)


        csv = csv[csv[['bkgApose', 'bkgStandardPose', 'bkgTpose', 'bkgFreePose',
                       'nobkgApose', 'nobkgStandardPose', 'nobkgTpose', 'nobkgFreePose']].notnull().any(1)]

        # filtering with query method
        csv.query('Age != "NaN" and Height != "NaN" and Weight != "NaN" and Gender != "NaN"', inplace = True)


        self.img_path = img_path

        self.age = csv['Age'].tolist()
        self.gender = csv['Gender'].tolist()
        self.heights = csv['Height'].tolist()
        self.weights = csv['Weight'].tolist()



        self.bkgApose = csv['bkgApose'].tolist()
        self.bkgStandardPose = csv['bkgStandardPose'].tolist()
        self.bkgTpose = csv['bkgTpose'].tolist()
        self.bkgFreePose = csv['bkgFreePose'].tolist()

        self.nobkgApose = csv['nobkgApose'].tolist()
        self.nobkgStandardPose = csv['nobkgStandardPose'].tolist()
        self.nobkgTpose = csv['nobkgTpose'].tolist()
        self.nobkgFreePose = csv['nobkgFreePose'].tolist()



        self.MidNeckCircumference = csv['MidNeckCircumference'].tolist()
        self.NeckCircumference1 = csv['NeckCircumference1'].tolist()
        self.ShoulderWidth = csv['ShoulderWidth'].tolist()
        self.SleeveCircumference = csv['SleeveCircumference'].tolist()
        self.ArmLength1 = csv['ArmLength-1'].tolist()
        self.BicepsCircumference = csv['BicepsCircumference'].tolist()
        self.WristCircumference = csv['WristCircumference'].tolist()
        self.BreastCircumference = csv['BreastCircumference'].tolist()

        self.UpperBreastCircumference = csv['UpperBreastCircumference'].tolist()
        self.LowerBreastCircumference = csv['LowerBreastCircumference'].tolist()
        self.WaistCircumference1 = csv['WaistCircumference1'].tolist()
        self.AbodomenCircumference = csv['AbodomenCircumference'].tolist()

        self.CoxaCircumference = csv['CoxaCircumference'].tolist()
        self.HipCircumference = csv['HipCircumference'].tolist()
        self.ThighCircumference = csv['ThighCircumference'].tolist()
        self.CalfCircumference = csv['CalfCircumference'].tolist()

        self.LegLength1 = csv['LegLength-1'].tolist()
        self.FrontNeck2HipLength1 = csv['FrontNeck2HipLength-1'].tolist()
        self.BackNeck2HipLength1 = csv['BackNeck2HipLength-1'].tolist()

        # plt.plot(self.ThighCircumference)
        # plt.plot(self.weights)
        # plt.plot(self.ThighCircumference)
        # plt.show()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([transforms.Grayscale(3),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomAffine(360, (0.1, 0.1), scale=(0.6, 1.0), shear=7),
                                             transforms.Resize(256),
                                             transforms.RandomCrop(256),

                                             # transforms.RandomPerspective(),
                                             transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                             transforms.ToTensor(),

                                             # transforms.RandomErasing(scale=(0.02, 0.08)),
                                             self.normalize
                                             ])



        self.model = gluoncv.model_zoo.get_model('deeplab_resnet152_voc', pretrained=True, ctx=ctx)

        self.cache_path = './cache/'
        os.makedirs(self.cache_path, exist_ok=True)




    def __getitem__(self, item):
        gender_value = torch.tensor(int(self.gender[item] == "male")).long()
        height_value = torch.tensor(float(self.heights[item])).float()
        weight_value = torch.tensor(float(self.weights[item])).float()
        age_value = torch.tensor(self.age[item]).float()



        # photo
        photos = []

        name = self.bkgApose[item]
        if str(name) != "nan":
            photos.append(name)
        name = self.bkgStandardPose[item]
        if str(name) != "nan":
            photos.append(name)
        name = self.bkgTpose[item]
        if str(name) != "nan":
            photos.append(name)
        name = self.bkgFreePose[item]
        if str(name) != "nan":
            photos.append(name)

        name = self.nobkgApose[item]
        if str(name) != "nan":
            photos.append(name)
        name = self.nobkgStandardPose[item]
        if str(name) != "nan":
            photos.append(name)
        name = self.nobkgTpose[item]
        if str(name) != "nan":
            photos.append(name)
        name = self.nobkgFreePose[item]
        if str(name) != "nan":
            photos.append(name)

        assert len(photos) != 0

        name = photos[np.random.randint(0, len(photos))]

        try:
            cache_path = self.cache_path + str(name) + '.png'
            if os.path.exists(cache_path):
                img = Image.open(cache_path)
            else:
                img_path = self.img_path + name
                img = Image.open(img_path).convert('RGB')
                img = transforms.Resize(256)(img)
                mask = segmentation_mask(self.model, img)
                img = np.array(img)
                img[~mask] = 0
                img = Image.fromarray(img)
                img.save(cache_path)

            img = self.transform(img)
        except:
            img = torch.zeros(3, 256, 256)
            img = self.normalize(img)


        # plt.imshow((img.transpose(0, 2).numpy() * 0.25) + 0.45)
        # plt.pause(1e-2)

        target = torch.tensor([self.MidNeckCircumference[item],
                               self.NeckCircumference1[item],
                               self.ShoulderWidth[item],
                               self.SleeveCircumference[item],

                               self.ArmLength1[item],
                               self.BicepsCircumference[item],
                               self.WristCircumference[item],
                               self.BreastCircumference[item],

                               self.UpperBreastCircumference[item],
                               self.LowerBreastCircumference[item],
                               self.WaistCircumference1[item],
                               self.AbodomenCircumference[item],

                               self.CoxaCircumference[item],
                               self.HipCircumference[item],
                               self.ThighCircumference[item],
                               self.CalfCircumference[item],

                               self.LegLength1[item],
                               self.FrontNeck2HipLength1[item],
                               self.BackNeck2HipLength1[item],

                               ]).float()

        return gender_value, height_value, weight_value, age_value, target, img

    def __len__(self):
        return len(self.gender)

    # self.LegLength1 = csv['LegLength-1'].tolist()
    # self.FrontNeck2HipLength1 = csv['FrontNeck2HipLength-1'].tolist()
    # self.BackNeck2HipLength1 = csv['BackNeck2HipLength-1'].tolist()


if __name__ == '__main__':

    # dataset = GlinsunDataset("/home/sparky/body-dataset/Overall-Body-Database-20201009.csv",
    #                          "/home/sparky/body-dataset/")
    dataset = GlinsunDataset("./Overall-Body-Database-20201017-corrected.csv",
                             "/home/sparky/Documents/Projects/aibody-dataset/")



    for x in dataset:
        pass
