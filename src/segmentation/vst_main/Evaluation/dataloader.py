from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        pred_names = os.listdir(pred_root)
        #label_names = os.listdir(label_root)

        if pred_root.split('/')[-2] == 'PASCAL-S':
            # remove the following image
            if '424.png' in pred_names:
                pred_names.remove('424.png')
            if '460.png' in pred_names:
                pred_names.remove('460.png')
            if '359.png' in pred_names:
                pred_names.remove('359.png')
            if '408.png' in pred_names:
                pred_names.remove('408.png')
            if '622.png' in pred_names:
                pred_names.remove('622.png')

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), pred_names))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), pred_names))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
