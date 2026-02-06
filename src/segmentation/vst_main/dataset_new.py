from PIL import Image
from torch.utils import data
import vst_main.transforms as trans
from torchvision import transforms
import random
import os
import os.path as op


def load_list(dataset_name, data_root):

    images = []
    labels = []
    contours = []

    img_root = data_root + dataset_name + '/DUTS-TR-Image/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img[:-4]+'.jpg')
        labels.append(img_root.replace('/DUTS-TR-Image/', '/DUTS-TR-Mask/') + img[:-4]+'.png')
        contours.append(img_root.replace('/DUTS-TR-Image/', '/DUTS-TR-Contour/') + img[:-4] + '.png')

    return images, labels, contours


def load_test_list(data_root):
    images = []
    class_names = os.listdir(data_root)
    for class_name in class_names:
        if 'DS_Store' in class_name:
            continue
        img_root = op.join(data_root, class_name)

        for img_name in os.listdir(img_root):
            if '.jpg' not in img_name:
                continue
            img_path = op.join(img_root, img_name)
            images.append(img_path)

    return images


def load_INbreast_test_list(data_root):
    images = []
    for img_name in os.listdir(data_root):
        if '.jpg' not in img_name:
            continue
        img_path = op.join(data_root, img_name)
        images.append(img_path)

    return images

def load_debug_list(data_root):
    images = []
    for img_name in os.listdir(data_root):
        if '.jpg' not in img_name:
            continue
        img_path = op.join(data_root, img_name)
        images.append(img_path)

    return images

class ImageData(data.Dataset):
    def __init__(self, test_img_class_root, data_root, transform, mode, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):

        if mode == 'train':
            self.image_path, self.label_path, self.contour_path = load_list(test_img_class_root, data_root)
        else:
            self.image_path = load_INbreast_test_list(test_img_class_root)
            # self.image_path = load_test_list(test_img_class_root)
            # self.image_path = load_debug_list(r'./test_demo')

        self.transform = transform
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            contour = Image.open(self.contour_path[item]).convert('L')
            random_size = self.scale_size

            new_img = trans.Scale((random_size, random_size))(image)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)
            new_contour = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(contour)

            # random crop
            w, h = new_img.size
            if w != self.img_size and h != self.img_size:
                x1 = random.randint(0, w - self.img_size)
                y1 = random.randint(0, h - self.img_size)
                new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_contour = new_contour.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)
                new_contour = new_contour.transpose(Image.FLIP_LEFT_RIGHT)

            new_img = self.transform(new_img)

            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)

            contour_14 = self.label_14_transform(new_contour)
            contour_28 = self.label_28_transform(new_contour)
            contour_56 = self.label_56_transform(new_contour)
            contour_112 = self.label_112_transform(new_contour)
            contour_224 = self.t_transform(new_contour)

            return new_img, label_224, label_14, label_28, label_56, label_112, \
                   contour_224, contour_14, contour_28, contour_56, contour_112
        else:

            image = self.transform(image)

            return image, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)

class ImageData_One(data.Dataset):
    def __init__(self, test_img_path, transform, mode, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):
        self.image_path = [test_img_path]
            # self.image_path = load_test_list(test_img_class_root)
            # self.image_path = load_debug_list(r'./test_demo')

        self.transform = transform
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])

        image = self.transform(image)

        return image, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':

        transform = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = trans.Compose([
            transforms.ToTensor(),
        ])
        label_14_transform = trans.Compose([
            trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_28_transform = trans.Compose([
            trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_56_transform = trans.Compose([
            trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_112_transform = trans.Compose([
            trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        scale_size = 256
    else:
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform, mode)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset


def get_loader_one(img_path, img_size, mode='test'):

    transform = trans.Compose([
        trans.Scale((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
    ])

    dataset = ImageData_One(img_path, transform, mode)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset