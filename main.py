# -*- coding: utf-8 -*-

import sys
import argparse
import random

import math
import numpy as np
import torch
import time
import os
import conf
from copy import deepcopy

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TRAINED = 0
SKIPPED = 1
FINISHED = 2


def get_path():
    path = 'log/'

    # information about used data type
    path += conf.args.dataset + '/'

    # information about used model type

    if conf.args.noisy_type:
        path += conf.args.method + "_noisy/"
    else:
        path += conf.args.method + '/'

    # information about domain(condition) of training data
    if conf.args.src == ['rest']:
        path += 'src_rest' + '/'
    elif conf.args.src == ['all']:
        path += 'src_all' + '/'
    elif conf.args.src is not None and len(conf.args.src) >= 1:
        path += 'src_' + '_'.join(conf.args.src) + '/'

    if conf.args.tgt:
        if conf.args.noisy_type:
            path += 'tgt_' + conf.args.tgt + '_{}'.format(conf.args.noisy_type) + \
                    '_{}'.format(conf.args.noisy_size) + '_{}/'.format(conf.args.noisy_class)
        else:
            path += 'tgt_' + conf.args.tgt + '/'

    if conf.args.noisy_type:
        if "FGSM" in conf.args.noisy_type:
            path += "{}/".format(conf.args.tta_attack_eps)

    if conf.args.tta_attack_type:
        path += "tta_attack_{}_num{}_step{}_eps{}/".format(conf.args.tta_attack_type, conf.args.tta_attack_num_samples,
                                                           conf.args.tta_attack_step, conf.args.tta_attack_eps)

    path += conf.args.log_prefix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path


def attack_online(learner, current_num_sample):
    if conf.args.method == "SRC_FOR_TTA_ATTACK":  # for ablation
        ret_val = learner.train_online(current_num_sample, add_memory=True, evaluation=False)  # add memory FIFO
        if ret_val == TRAINED:  # if current_num_sample % update_every_x == 0
            learner.reset_purturb()  # reset pruturb to zero
            for step in range(conf.args.tta_attack_step):
                learner.tta_attack_train(current_num_sample, step)  # adding generated attack samples

            learner.tta_attack_update(current_num_sample)
            ret_val = learner.train_online(current_num_sample, add_memory=False, evaluation=False)  # train online

    else:  # normal
        if current_num_sample % conf.args.update_every_x == 0:
            temp_model_state = deepcopy(learner.net.state_dict())  # save model state before train

        ret_val = learner.train_online(current_num_sample, add_memory=True, evaluation=False)  # add memory FIFO

        if ret_val == TRAINED:
            assert current_num_sample % conf.args.update_every_x == 0
            learner.reset_purturb()  # reset pruturb to zero
            learner.net.load_state_dict(temp_model_state)  # reset model
            for step in range(conf.args.tta_attack_step):
                learner.tta_attack_train(current_num_sample, step)  # adding generated attack samples
                learner.net.load_state_dict(temp_model_state)  # reset model for next testing step

            learner.tta_attack_update(current_num_sample)
            ret_val = learner.train_online(current_num_sample, add_memory=False, evaluation=False)  # train online

    return ret_val


def main():
    ######################################################################
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################### Hyperparameters #################
    if 'cifar100' in conf.args.dataset:
        opt = conf.CIFAR100Opt
    elif 'cifar10' in conf.args.dataset:
        opt = conf.CIFAR10Opt
    elif 'imagenet' in conf.args.dataset:
        opt = conf.IMAGENET_C
    else:
        raise NotImplementedError

    conf.args.opt = opt
    if conf.args.lr:
        opt['learning_rate'] = conf.args.lr
    if conf.args.weight_decay:
        opt['weight_decay'] = conf.args.weight_decay

    model = None

    if conf.args.model == "resnet18":
        from models import ResNet
        model = ResNet.ResNet18()
    elif conf.args.model == "resnet18_pretrained":
        import torchvision
        model = torchvision.models.resnet18(pretrained=True)
    elif conf.args.model == "resnet50":
        from models import ResNet
        model = ResNet.ResNet50()
    elif conf.args.model == "resnet50_pretrained":
        import torchvision
        model = torchvision.models.resnet50(pretrained=True)

    # import modules after setting the seed
    from data_loader import data_loader as data_loader
    from learner.dnn import DNN
    from learner.bn_stats import BN_Stats
    from learner.pseudo_label import PseudoLabel
    from learner.tent import TENT
    from learner.sotta import SoTTA
    from learner.cotta import CoTTA
    from learner.lame import LAME
    from learner.sar import SAR
    from learner.rotta import RoTTA
    from learner.eata import EATA
    from learner.src_for_tta_attack import SRC_FOR_TTA_ATTACK

    result_path, checkpoint_path, log_path = get_path()

    ########## Dataset loading ############################

    if conf.args.method == 'Src':
        learner_method = DNN
    elif conf.args.method == 'BN_Stats':
        learner_method = BN_Stats
    elif conf.args.method == 'PseudoLabel':
        learner_method = PseudoLabel
    elif conf.args.method == 'TENT':
        learner_method = TENT
    elif conf.args.method == 'CoTTA':
        learner_method = CoTTA
    elif conf.args.method == 'LAME':
        learner_method = LAME
    elif conf.args.method == 'SAR':
        learner_method = SAR
    elif conf.args.method == "RoTTA":
        learner_method = RoTTA
    elif conf.args.method == 'SoTTA':
        learner_method = SoTTA
    elif conf.args.method == "SRC_FOR_TTA_ATTACK":
        learner_method = SRC_FOR_TTA_ATTACK
    elif conf.args.method == "EATA":
        learner_method = EATA
    else:
        raise NotImplementedError

    since = time.time()

    print('##############Source Data Loading...##############')
    source_data_loader = data_loader.domain_data_loader(conf.args.dataset, conf.args.src,
                                                        conf.args.opt['file_path'],
                                                        batch_size=conf.args.opt['batch_size'],
                                                        valid_split=0,  # to be used for the validation
                                                        test_split=0, is_src=True)

    print('##############Target Data Loading...##############')
    target_data_loader = data_loader.domain_data_loader(conf.args.dataset, conf.args.tgt,
                                                        conf.args.opt['file_path'],
                                                        batch_size=conf.args.opt['batch_size'],
                                                        valid_split=0,
                                                        test_split=0, is_src=False)

    learner = learner_method(model, source_dataloader=source_data_loader,
                             target_dataloader=target_data_loader, write_path=log_path)

    time_elapsed = time.time() - since
    print('Data Loading Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #################### Training #########################

    since = time.time()

    # make dir if doesn't exist
    if not os.path.exists(result_path):
        oldumask = os.umask(0)
        os.makedirs(result_path, 0o777)
        os.umask(oldumask)
    if not os.path.exists(checkpoint_path):
        oldumask = os.umask(0)
        os.makedirs(checkpoint_path, 0o777)
        os.umask(oldumask)

    if not conf.args.online:

        start_epoch = 1
        best_acc = -9999
        best_epoch = -1

        for epoch in range(start_epoch, conf.args.epoch + 1):
            learner.train(epoch)

        learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
        learner.dump_eval_online_result(is_train_offline=True)  # eval with final model

        time_elapsed = time.time() - since
        print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    else:  # online
        current_num_sample = 1
        num_sample_end = conf.args.nsample
        best_acc = -9999
        best_epoch = -1

        finished = False

        while not finished and current_num_sample < num_sample_end:
            if conf.args.tta_attack_type:
                ret_val = attack_online(learner, current_num_sample)
            else:
                ret_val = learner.train_online(current_num_sample)

            if ret_val == FINISHED:
                break
            elif ret_val == SKIPPED:
                pass
            elif ret_val == TRAINED:
                pass
            current_num_sample += 1

        if not conf.args.remove_cp:
            learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                    checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
        learner.dump_eval_online_result()

        time_elapsed = time.time() - since
        print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    if conf.args.remove_cp:
        last_path = checkpoint_path + 'cp_last.pth.tar'
        best_path = checkpoint_path + 'cp_best.pth.tar'
        try:
            os.remove(last_path)
            os.remove(best_path)
        except Exception:
            pass


def parse_arguments():
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to be used, in [cifar10noisy, cifar100noisy, imagenetnoisy]')
    parser.add_argument('--model', type=str, default='',
                        help='Base model to use')
    parser.add_argument('--method', type=str, default='',
                        help='TTA method')
    parser.add_argument('--src', nargs='*', default=None,
                        help='Specify source domains; not passing an arg load default src domains specified in conf.py')
    parser.add_argument('--tgt', type=str, default=None,
                        help='specific target domain; give "src" if you test under src domain')
    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    # Noisy data settings
    parser.add_argument('--noisy_type', default=None, type=str,
                        help='Noisy test data : divide, repeat, oneclassrepeat, cifar100, cifar100c, gaussian, uniform, attack')
    parser.add_argument('--noisy_size', default=None, type=int,
                        help='Noisy test data size')
    parser.add_argument('--noisy_class', default=None, nargs="*", type=int,
                        help='Noisy test data target class')

    # Additional
    parser.add_argument('--dummy', action='store_true', default=False,
                        help='do nothing')
    parser.add_argument('--log_prefix', type=str, default='',
                        help='Prefix of log file path')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--remove_cp', action='store_true',
                        help='Remove checkpoints after evaluation')

    # Learning parameters
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight_decay to overwrite conf.py')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--use_learned_stats', action='store_true',
                        help='Use learned stats uses exponential moving average with bn_momentum')
    parser.add_argument('--bn_momentum', type=float, default=0.1,
                        help='Momentum of exponential moving average')
    parser.add_argument('--epoch', type=int, default=1,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--online', action='store_true',
                        help='training via online learning?')
    parser.add_argument('--update_every_x', type=int, default=1,
                        help='number of target samples used for every update')
    parser.add_argument('--memory_type', type=str, default='FIFO',
                        help='Memory type. FIFO as default')
    parser.add_argument('--memory_size', type=int, default=1,
                        help='number of previously trained data to be used for training')

    # Data
    parser.add_argument('--train_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for train')
    parser.add_argument('--valid_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for valid')
    parser.add_argument('--test_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for test')
    parser.add_argument('--nsample', type=int, default=20000000,
                        help='How many samples do you want use for train')
    parser.add_argument('--tgt_train_dist', type=int, default=1,
                        help='0: real selection'
                             '1: random selection'
                             '4: dirichlet distribution'
                             '5: no shuffling'
                             '6: online imbalanced label shifts (ImageNet-C only; details in SAR)'
                        )
    parser.add_argument('--dirichlet_beta', type=float, default=0.1,
                        help='the concentration parameter of the Dirichlet distribution for heterogeneous partition.')
    parser.add_argument('--validation', action='store_true',
                        help='Use validation data instead of test data for hyperparameter tuning')

    # LAME
    parser.add_argument('--parallel', type=bool, default=False)

    # CoTTA
    parser.add_argument('--ema_factor', type=float, default=0.999)
    parser.add_argument('--restoration_factor', type=float, default=0.01)
    parser.add_argument('--aug_threshold', type=float, default=0.92)

    # SoTTA
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for HLoss')
    parser.add_argument('--loss_scaler', type=float, default=0,
                        help='loss_scaler for entropy_loss')
    parser.add_argument('--esm', action='store_true', default=False,
                        help='changes Adam to ESM+Adam')
    parser.add_argument('--high_threshold', default=0.99, type=float,
                        help='High confidence threshold')
    parser.add_argument('--log_grad', action='store_true', default=False,
                        help="log grad-norm")

    # EATA settings
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\\epsilon in Eqn. (5) for filtering redundant samples')

    # TTA Attack setting
    parser.add_argument('--tta_attack_eps', default=8 / 255, type=float,
                        help='eps of attack, ie 8/255 for FGSM')
    parser.add_argument('--tta_attack_type', default=None, type=str,
                        help='adversarial attack : targeted , indiscriminate, stealthy')
    parser.add_argument('--tta_attack_step', default=200, type=int,
                        help='number of iteration for training attack data')
    parser.add_argument('--tta_attack_num_samples', default=20, type=int,
                        help='number of attack samples')
    parser.add_argument('--tta_attack_target', default=None, type=int,
                        help='target attack class')

    return parser.parse_args()


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('Command:', end='\t')
    print(" ".join(sys.argv))
    conf.args = parse_arguments()
    print(conf.args)
    set_seed()
    main()
