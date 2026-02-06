import os
import torch
import Training
import Testing
from Evaluation import main
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--data_root', default='./Data/', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=11, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=30000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=45000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='DUTS/DUTS-TR', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='./pretrained_model/', type=str, help='save model path')

    # test
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='/data2/machong/projects/VST-main/INbreast_Saliency/0_enhance/test/', type=str, help='save saliency maps path')
    # parser.add_argument('--save_test_path_root', default='/data2/machong/projects/VST-main/INbreast_Saliency/0_enhance/train/', type=str, help='save saliency maps path')
    # parser.add_argument('--save_test_path_root', default='/data2/machong/projects/VST-main/FIGRIM_Saliency/Targets/', type=str, help='save saliency maps path')
    # parser.add_argument('--save_test_path_root', default='/data2/machong/projects/VST-main/FIGRIM_Saliency/Targets/', type=str, help='save saliency maps path')
    # parser.add_argument('--save_test_path_root', default='/data2/machong/projects/VST-main/FIGRIM_Saliency/Fillers/', type=str, help='save saliency maps path')
    # parser.add_argument('--save_test_path_root', default='/data/machong/eye_tools/VST-main/demo_save/', type=str, help='save saliency maps path')
    # parser.add_argument('--test_paths', type=str, default='DUTS/DUTS-TE+ECSSD+HKU-IS+PASCAL-S+DUT-O+BSD')
    parser.add_argument('--test_paths', type=str, default="/data2/machong/dataset/INbreast-Gaze/Mass_NBM_labelme_cut_img/0302/0_enhance/test/")
    # parser.add_argument('--test_paths', type=str, default="/data2/machong/dataset/INbreast-Gaze/Mass_NBM_labelme_cut_img/0302/0_enhance/train/")
    # parser.add_argument('--test_paths', type=str, default='/data2/machong/dataset/FIGRIM_Fixation_Dataset/SRC/Targets/Targets/')
    # parser.add_argument('--test_paths', type=str, default='/data2/machong/dataset/FIGRIM_Fixation_Dataset/SRC/Fillers/Fillers/')
    # parser.add_argument('--test_paths', type=str, default='/data/machong/eye_tools/VST-main/demo_save')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGB_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "8"

    num_gpus = torch.cuda.device_count()
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
    if args.Evaluation:
        main.evaluate(args)