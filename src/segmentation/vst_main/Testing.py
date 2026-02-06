import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
# from .dataset import get_loader
from vst_main.dataset_new import get_loader, get_loader_one
import vst_main.transforms as trans
from torchvision import transforms
import time
from vst_main.Models.ImageDepthNet import ImageDepthNet
from torch.utils import data
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse
from PIL import Image


def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = args.save_model_dir + 'RGB_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(model_path))

    # load model
    # net.load_state_dict(torch.load(model_path))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(model_path))

    test_dataset = get_loader(args.test_paths, args.data_root, args.img_size, mode='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

    time_list = []
    for data_batch in tqdm(test_loader):
        images, image_w, image_h, image_path = data_batch
        images = Variable(images.cuda())

        starts = time.time()
        outputs_saliency, outputs_contour = net(images)
        ends = time.time()
        time_use = ends - starts
        time_list.append(time_use)

        mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

        image_w, image_h = int(image_w[0]), int(image_h[0])

        output_s = F.sigmoid(mask_1_1)

        output_s = output_s.data.cpu().squeeze(0)

        transform = trans.Compose([
            transforms.ToPILImage(),
            trans.Scale((image_w, image_h))
        ])
        output_s = transform(output_s)

        numpy_output = np.array(output_s)
        resize_output = cv2.resize(np.array(output_s), (512, 512))

        img_name = image_path[0].split('/')[-1]
        class_name = image_path[0].split('/')[-2]

        save_numpy_root = op.join(args.save_test_path_root, 'saliency_array', class_name)
        os.makedirs(save_numpy_root, exist_ok=True)
        np.save(op.join(save_numpy_root, img_name.replace('.jpg', '.npy')), numpy_output)

        save_hm_img_root = op.join(args.save_test_path_root, 'saliency_img', class_name)
        os.makedirs(save_hm_img_root, exist_ok=True)

        plt.imshow(cv2.resize(cv2.imread(image_path[0]), (512, 512)))
        plt.axis('off')

        plt.imshow(resize_output, alpha=0.5, cmap=plt.cm.jet)
        plt.axis('off')
        plt.savefig(op.join(save_hm_img_root, img_name), bbox_inches='tight', dpi=300, pad_inches=0)
        plt.clf()


def test_img(net, img_path):
    transform = trans.Compose([
        trans.Scale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
    ])

    if isinstance(img_path, str):
        image = Image.open(img_path).convert('RGB')
    else:
        image = Image.fromarray(img_path.astype('uint8')).convert('RGB')
    image_w, image_h = int(image.size[0]), int(image.size[1])
    image = transform(image)
    images = image.unsqueeze(0)

    images = Variable(images.cuda())

    outputs_saliency, outputs_contour = net(images)

    mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

    image_w, image_h = int(image_w), int(image_h)

    output_s = F.sigmoid(mask_1_1)

    output_s = output_s.data.cpu().squeeze(0)

    transform = trans.Compose([
        transforms.ToPILImage(),
        trans.Scale((image_w, image_h))
    ])
    output_s = transform(output_s)

    numpy_output = np.array(output_s)

    return numpy_output



def VST_test_once(img_path):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--train_steps', default=60000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=11, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=30000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=45000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='DUTS/DUTS-TR', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='../src/segmentation/vst_main/pretrained_model/', type=str, help='save model path')

    # test
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGB_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    # define model
    cudnn.benchmark = True
    net = ImageDepthNet(args)
    net.cuda()
    net.eval()
    # load model (multi-gpu)
    #print('Current directory:', os.getcwd())

    model_path = args.save_model_dir + 'RGB_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    # print('Model loaded from {}'.format(model_path))

    vst_mask = test_img(net, img_path)

    return vst_mask
