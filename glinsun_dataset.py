import torch
import torch.utils.data as data
import mxnet
import cv2
import pandas as pd
import numpy as np
import scipy
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import gluoncv
import mxnet as mx
import os
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.data.transforms.presets.yolo import transform_test
# 588,female,50,149.5,5740%,rectangle,2748-0671_1.png,523-0671_1.png,NaN,NaN,NaN,NaN,NaN,NaN,34,38,40,49,47,30,17,94,92.5,93,87,88,93,95,56,34.5,90,59,55,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
from PIL import ImageOps

ctx = mx.cpu()


def segmentation_mask(seg_model, box_model, img):
    img = mxnet.nd.array(np.asarray(img))
    img = transform_test(img)
    ndimg = img[0]
    npimg = img[1]
    class_IDs, scores, bounding_boxs = box_model(ndimg)
    box = bounding_boxs.asnumpy().squeeze()[((class_IDs.asnumpy()) == 14).squeeze()]
    scores = scores.asnumpy().squeeze()[((class_IDs.asnumpy()) == 14).squeeze()]

    if box.shape[0] == 0:
        raise Exception('No human detected')

    box = box[scores.argmax()].astype(np.int)

    x0 = box[0]
    y0 = box[1]
    x1 = box[2]
    y1 = box[3]

    npimg = npimg[y0:y1, x0:x1]

    # plt.imshow(npimg)
    # plt.show()

    # using cpu
    img = mxnet.nd.array(np.asarray(npimg))
    img = test_transform(img, ctx)
    output = seg_model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy() == 15.

    # from gluoncv.utils.viz import get_color_pallete
    # mask = get_color_pallete(predict, 'pascal_voc')
    # plt.imshow(predict)
    # plt.show()

    npimg[~predict] = 0

    return npimg


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


class GlinsunDataset(data.Dataset):

    def __init__(self, csv_file, img_path, to_generate=False):
        self.to_generate = to_generate
        csv = pd.read_csv(csv_file, encoding="GBK")
        # csv = csv.replace(np.nan, '', regex=True)
        # csv.query('bkgFreePose != "NaN"', inplace=True)

        csv = csv[csv[['bkgApose', 'bkgStandardPose', 'bkgTpose', 'bkgFreePose',
                       'nobkgApose', 'nobkgStandardPose', 'nobkgTpose', 'nobkgFreePose']].notnull().any(1)]

        csv = csv[csv[['body_shape_manual']].notnull().all(1)]

        # filtering with query method
        csv.query('Age != "NaN" and Height != "NaN" and Weight != "NaN" and Gender != "NaN"', inplace=True)

        self.img_path = img_path

        self.age = csv['Age'].tolist()
        self.gender = csv['Gender'].tolist()
        self.heights = csv['Height'].tolist()
        self.weights = csv['Weight'].tolist()

        self.bodyShape = csv['body_shape_manual'].tolist()

        self.Videos = csv['Video'].tolist()

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
        # add scale
        self.transform = transforms.Compose([
            transforms.RandomGrayscale(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(360),
            transforms.RandomPerspective(distortion_scale=0.1),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),

            transforms.RandomErasing(scale=(0.02, 0.08)),
            self.normalize
        ])

        self.seg_model = gluoncv.model_zoo.get_model('deeplab_resnet152_voc', pretrained=True, ctx=ctx)
        self.box_model = gluoncv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)

        self.cache_path = './cache/'
        os.makedirs(self.cache_path, exist_ok=True)




    def __getitem__(self, item):
        gender_value = torch.tensor(int((self.gender[item] == "male") or (self.gender[item] == "Male"))).long()
        height_value = torch.tensor(float(self.heights[item])).float()
        weight_value = torch.tensor(float(self.weights[item])).float()
        age_value = torch.tensor(self.age[item]).float()

        # try:
        shape_value = torch.tensor(convert_shape(self.gender[item], self.bodyShape[item])).long()
        # except:
        #     print('fail')

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
        video_path = self.Videos[item]

        use_video = np.random.ranf() > 0.5 and str(video_path) != "nan"

        try:
            if use_video:
                cap = cv2.VideoCapture(self.img_path + video_path) #'video01-yangluo/0750.mp4')
                amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frame_id = np.random.randint(0, amount_of_frames)
                cache_path = self.cache_path + str(video_path.split('/')[1]) + 'frame_id:' + str(frame_id) + '.png'
            else:
                cache_path = self.cache_path + str(name)

            if os.path.exists(cache_path):
                img = Image.open(cache_path)
            else:
                if not self.to_generate:
                    raise Exception('no cache')

                if use_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    res, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame).convert('RGB')
                else:
                    img_path = self.img_path + name
                    frame = Image.open(img_path).convert('RGB')

                img = transforms.Resize(512)(frame)
                img = segmentation_mask(self.seg_model, self.box_model, img)
                img = Image.fromarray(img)


                desired_size = max(img.height, img.width)
                delta_w = desired_size - img.width
                delta_h = desired_size - img.height
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                new_im = ImageOps.expand(img, padding)
                # new_im.show()
                new_im.save(cache_path)
                img = new_im

            img = self.transform(img)
            # plt.imshow((img.transpose(0, 2).numpy() * 0.25) + 0.45)
            # plt.show()


        except Exception as e:
            if self.to_generate:
                print(e)
            img = torch.zeros(3, 256, 256)
            img = self.normalize(img)

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

        return shape_value, gender_value, height_value, weight_value, age_value, target, img

    def __len__(self):
        return len(self.gender)

    # self.LegLength1 = csv['LegLength-1'].tolist()
    # self.FrontNeck2HipLength1 = csv['FrontNeck2HipLength-1'].tolist()
    # self.BackNeck2HipLength1 = csv['BackNeck2HipLength-1'].tolist()


if __name__ == '__main__':

    # dataset = GlinsunDataset("/home/sparky/body-dataset/Overall-Body-Database-20201009.csv",
    #                          "/home/sparky/body-dataset/")
    dataset = GlinsunDataset("./Overall-Body-Database-20201028.csv", "/home/sparky/body-dataset/", True)
                             # "/home/sparky/Documents/Projects/aibody-dataset/")
    while True:
        for x in dataset:
            pass
