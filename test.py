from __future__ import division
from __future__ import print_function

import os, sys
import shutil
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.append('./FewShotLearning')

# from options import parse_args
from FewShotLearning.report import Tap, create_result_subdir
from FewShotLearning.dataloader.data_manager import FewShotDataManager
from FewShotLearning.methods.protonet import ProtoNet_Sinkhorn

def parse_args():
    parser = argparse.ArgumentParser(description= 'few-shot params')

    parser.add_argument('--base_data_path', default='/data_disk/dataset/few_shot_learning', type=str, help='')

    parser.add_argument('--model',  default='ResNet12', help='model: ResNet12/10')
    parser.add_argument('--test_dataset_name',  default='cub', help='miniImagenet/cub/tieredImagenet/cars')
    parser.add_argument('--image_size', default=112, type=int, help='84 or 112')
    parser.add_argument('--feature_scale', default=2, type=int, help='1 or 2, different number of features, 1 means 7/5, 2 means 14/10')
    parser.add_argument('--gpu_devices', default='1', type=str)

    parser.add_argument('--seed', default=0, type=int, help='radnom seed for initialization')
    parser.add_argument('--test_seed', default=111, type=int, help='radnom seed for load test tasks')

    parser.add_argument('--test_n_way'  ,  default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--test_n_shot',   default=1, type=int,  help='number of labeled data in each class, same as n_support')
    parser.add_argument('--test_n_query',  default=15, type=int,  help='number of query data in each class')
    parser.add_argument('--test_epoch_size',  default=2000,  type=int, help='how many episodes in one testing epoch')
    parser.add_argument('--fewshot_test_batch',  default=4, type=int, help='how many episodes in one testing batch')

    parser.add_argument('--trained_model_path', default='/data_disk/deeplearning/trained_model/single_domain/miniImagenet/5-shot/ResNet12-miniImagenet-5shot.pth', type=str, help='')

    parser.add_argument('--drop_rate',      default=0.0,   type=float, help="drop_rate")
    parser.add_argument('--drop_block',     default=False, type=bool, help='whether to use drop block instead dropout')
    parser.add_argument('--dropblock_size', default=5,     type=int, help="dropblock_size")

    parser.add_argument('--thresh', default=0.01, type=float, help='iteration stop condition')
    parser.add_argument('--maxiters', default=1000, type=int, help='max number of iterations')
    parser.add_argument('--lam', default=80, type=int, help='lambda')


    return parser.parse_args()

if __name__ == "__main__":

    ######### load settings ##########
    args = parse_args()

    for arg in vars(args):
        print('{}--{}'.format(arg, getattr(args, arg)))

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus
    print("Currently using {} GPUs, {}".format(num_gpus, args.gpu_devices))  # using only one gpu to train

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True  # speed up training
    cudnn.deterministic = True

    test_data_path = os.path.join(args.base_data_path, args.test_dataset_name, 'novel.json')
    fewshot_data_manager_test = FewShotDataManager(test_data_path, args.image_size, args.test_n_way, args.test_n_shot,
                                                    args.test_n_query, args.test_epoch_size, args.fewshot_test_batch, args.test_seed)

    model = ProtoNet_Sinkhorn(args)

    if args.trained_model_path:
        model.load_model(args.trained_model_path)
        # model.test_loop(-1, fewshot_data_manager_val.data_loader, 'Val')
        model.test_loop(fewshot_data_manager_test.data_loader)
