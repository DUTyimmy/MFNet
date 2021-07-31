import os
import torch
import PIL.Image
import numpy as np
from torch.utils import data
import xml.etree.ElementTree as ET


class TorchvisionNormalize(object):
    def __init__(self, mean=(0.447, 0.407, 0.386), std=(0.244, 0.250, 0.253)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class MySalTrainData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, resize=256, infer=False, transform=False, stage=1):
        super(MySalTrainData, self).__init__()
        self.root = root
        self._transform = transform
        self.infer = infer
        self.resize = resize
        self.stage = stage

        img_root = os.path.join(self.root, 'duts-train/image')
        lbl1_root = os.path.join(self.root, 'pseudo_labels/label0_' + str(self.stage))
        lbl2_root = os.path.join(self.root, 'pseudo_labels/label1_' + str(self.stage))
        file_names = os.listdir(img_root)

        self.img_names = []
        self.lbl1_names = []
        self.lbl2_names = []

        for i, name in enumerate(file_names):

            self.lbl1_names.append(os.path.join(lbl1_root, name[:-4] + '.png'))
            self.lbl2_names.append(os.path.join(lbl2_root, name[:-4] + '.png'))
            self.img_names.append(os.path.join(img_root, name))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = img.convert('RGB')
        img = img.resize((self.resize, self.resize))
        img = np.array(img, dtype=np.uint8)

        lbl1_file = self.lbl1_names[index]
        lbl1 = PIL.Image.open(lbl1_file)
        lbl1 = lbl1.convert('L')
        lbl1 = lbl1.resize((self.resize, self.resize))
        lbl1 = np.array(lbl1)

        lbl2_file = self.lbl2_names[index]
        lbl2 = PIL.Image.open(lbl2_file)
        lbl2 = lbl2.convert('L')
        lbl2 = lbl2.resize((self.resize, self.resize))
        lbl2 = np.array(lbl2)

        if self._transform:
            return self.transform(img, lbl1, lbl2)
        else:
            return img, lbl1, lbl2

    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img, lbl1, lbl2):

        img = img.astype(np.float64)/255.0
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        lbl1 = lbl1.astype(np.float32)/255.0
        lbl1 = torch.from_numpy(lbl1).float()

        lbl2 = lbl2.astype(np.float32)/255.0
        lbl2 = torch.from_numpy(lbl2).float()

        return img, lbl1, lbl2


class MySalTestData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, resize=256, transform=False):
        super(MySalTestData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize
        self.img_names = []
        self.names = []

        file_names = os.listdir(self.root)

        for i, name in enumerate(file_names):
            ifpic = name.endswith('.jpg') or name.endswith('.png')
            if not ifpic:
                continue
            self.img_names.append(os.path.join(self.root, name))
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = img.convert('RGB')
        img_size = img.size
        img = img.resize((self.resize, self.resize))
        img = np.array(img, dtype=np.uint8)

        if self._transform:
            return self.transform(img), self.names[index], img_size
        else:
            return img, self.names[index], img_size

    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img):
        img = img.astype(np.float64)/255.0
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img


class MySalInferData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, resize=256, transform=False):
        super(MySalInferData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize
        img_root = os.path.join(self.root, 'duts-train/image')

        file_names = os.listdir(img_root)
        self.img_names = []

        self.img_names = []
        self.names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = img.convert('RGB')
        img_size = img.size
        img = img.resize((self.resize, self.resize))
        img = np.array(img, dtype=np.uint8)

        if self._transform:
            return self.transform(img), self.names[index], img_size
        else:
            return img, self.names[index], img_size

    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img):
        img = img.astype(np.float64)/255.0
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img


class MySalValData(data.Dataset):

    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, resize=256, transform=False):
        super(MySalValData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize
        self.img_names = []
        self.names = []

        img_root = os.path.join(self.root, 'ECSSD/image')
        file_names = os.listdir(img_root)

        for i, name in enumerate(file_names):
            ifpic = name.endswith('.jpg') or name.endswith('.png')
            if not ifpic:
                continue
            self.img_names.append(os.path.join(img_root, name))
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = img.convert('RGB')
        img_size = img.size
        img = img.resize((self.resize, self.resize))
        img = np.array(img, dtype=np.uint8)

        if self._transform:
            return self.transform(img), self.names[index], img_size
        else:
            return img, self.names[index], img_size

    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img):
        img = img.astype(np.float64)/255.0
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img


class MyCamTrainData(data.Dataset):
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    # cls2idx = dict(zip(os.listdir(self.root), list(range(1, len(os.listdir(self.root) + 1)))))

    def __init__(self, root, transform=False, resize=256):
        super(MyCamTrainData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize
        self.img_list = []
        self.cls_list = []
        self.root = os.path.join(self.root, 'dutscls-44')

        self.cls = os.listdir(self.root)
        self.cls2idx = dict(zip(self.cls, list(range(1, len(self.cls) + 1))))

        for cls_idx in self.cls:
            img_names = os.listdir(os.path.join(self.root, cls_idx))
            for img_name in img_names:
                self.img_list.append(img_name)
                self.cls_list.append([cls_idx])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        lbl_cls = self.cls_list[index][0]
        # lbl_idx = str(self.cls_list[index])
        # lbl_idx = lbl_idx.pop(1, -1)
        # print(lbl_idx)
        img_file, lbl = self.img_list[index], self.cls2idx[lbl_cls]
        img = PIL.Image.open(os.path.join(self.root, str(lbl_cls), img_file)).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img = np.array(img)

        onehot = np.zeros(len(self.cls))
        lbl = np.array(lbl) - 1
        onehot[lbl] = 1

        if self._transform:
            return self.transform(img, onehot)
        else:
            return img, onehot

    def transform(self, img, lbl):
        img = img.astype(np.float64) / 255.0
        lbl = lbl.astype(np.float32)
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify #256*256*3 to 3*256*256
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl


class MyCamInferData(data.Dataset):
    """
    load data in a folder
    """

    def __init__(self, root, scales, transform=True, img_normal=TorchvisionNormalize()):
        super(MyCamInferData, self).__init__()
        self.if_transform = transform
        self.scales = scales
        self.img_normal = img_normal

        img_root = os.path.join(root, 'duts-train/image')
        # img_root = os.path.join('test')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []

        for i, name in enumerate(file_names):
            ifpic = name.endswith('.jpg') or name.endswith('.png')
            if not ifpic:
                continue
            self.img_names.append(os.path.join(img_root, name))
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img = PIL.Image.open(self.img_names[index])
        img = img.convert('RGB')
        img_size = img.size
        # img = img.resize((self.resize, self.resize))
        img = np.array(img, dtype=np.uint8)

        mul_img_list = []
        mul_img_norm_list = []
        i = 0
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = self.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = self.trans(s_img)
            mul_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))  # why here -> flip to CAM

        if self.if_transform:
            for _ in self.scales:
                mul_img_norm_list.append(mul_img_list[0])
                i += 1
            return mul_img_norm_list, self.names[index], img_size
        else:
            return mul_img_list, self.names[index], img_size

    @staticmethod
    def trans(img):
        return np.transpose(img, (2, 0, 1))

    def pil_rescale(self, img, scale, order):
        height, width = img.shape[:2]
        target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
        return self.pil_resize(img, target_size, order)

    @staticmethod
    def pil_resize(img, size, order):
        resample = PIL.Image.BICUBIC
        if size[0] == img.shape[0] and size[1] == img.shape[1]:
            return img
        if order == 3:
            resample = PIL.Image.BICUBIC
        elif order == 0:
            resample = PIL.Image.NEAREST

        return np.asarray(PIL.Image.fromarray(img).resize(size[::-1], resample))


class ImageNetClsData(data.Dataset):
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, transform=False, resize=256):
        super(ImageNetClsData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize
        txts = os.listdir(os.path.join(self.root, 'data/det_lists'))
        txts = filter(lambda x: x.startswith('train_pos') or x.startswith('train_part'), txts)
        file2lbl = {}
        for txt in txts:
            files = open(os.path.join(self.root, 'data/det_lists', txt)).readlines()
            for f in files:
                f = f.strip('\n')+'.JPEG'
                if f in file2lbl:
                    file2lbl[f] += [int(txt.split('.')[0].split('_')[-1])]
                else:
                    file2lbl[f] = [int(txt.split('.')[0].split('_')[-1])]
        self.file2lbl = file2lbl.items()
        self.file2lbl = list(self.file2lbl)

    def __len__(self):
        return len(self.file2lbl)

    def __getitem__(self, index):
        # load image

        # print(len(self.file2lbl))
        img_file, lbl = self.file2lbl[index]
        img = PIL.Image.open(os.path.join(self.root, 'ILSVRC2014_DET_train', img_file)).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img = np.array(img)
        onehot = np.zeros(200)
        lbl = np.array(lbl)-1
        onehot[lbl] = 1

        if self._transform:
            return self.transform(img, onehot)
        else:
            return img, onehot

    def transform(self, img, lbl):
        img = img.astype(np.float64)/255.0
        lbl = lbl.astype(np.float32)
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify #256*256*3 to 3*256*256
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl


class MarineClsData(data.Dataset):
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    @staticmethod
    def cls_from_xml(xml_root):
        name2index = {'starfish': 0, 'holothurian': 1, 'echinus': 2, 'scallop': 3, 'waterweeds': 4}
        cls, index = [], []

        root = ET.parse(xml_root).getroot()
        children_node = root.getchildren()
        for child in children_node:
            if child.tag == 'object':
                child_node = child.getchildren()
                for childd in child_node:
                    if childd.tag == 'name':
                        cls.append(childd.text)
        cls = list(set(cls))
        for item in cls:
            index.append(name2index[item])

        onehot = np.zeros(5)
        if not len(index) == 0:
            onehot[np.array(index)] = 1
        return onehot

    def __init__(self, root, transform=False, resize=256):
        super(MarineClsData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize

        self.anno_root = os.path.join(self.root, 'Annotations')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.img_list, self.cls_list = [], []

        files = os.listdir(self.img_root)

        for file in files:
            xml_root = os.path.join(self.anno_root, file[:-4] + '.xml')
            cls = self.cls_from_xml(xml_root)

            self.img_list.append(file)
            self.cls_list.append(cls)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # load image

        # print(len(self.file2lbl))
        img_file, lbl = self.img_list[index], self.cls_list[index]
        img = PIL.Image.open(os.path.join(self.root, 'JPEGImages', img_file)).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img = np.array(img)

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img.astype(np.float64)/255.0
        lbl = lbl.astype(np.float32)
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify #256*256*3 to 3*256*256
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

