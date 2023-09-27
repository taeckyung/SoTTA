from copy import deepcopy

import conf
from utils import memory
from utils.loss_functions import *
from .dnn import DNN

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class SRC_FOR_TTA_ATTACK(DNN):
    def __init__(self, *args, **kwargs):
        super(SRC_FOR_TTA_ATTACK, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)  # required for evaluation
        self.mem_state = self.mem.save_state_dict()
        self.grad_norm_list = []
        self.original_model = deepcopy(self.net)

    def train_online(self, current_num_sample, add_memory=True, evaluation=True):
        """
        Train the model
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, 'previous_train_loss'):
            self.previous_train_loss = 0

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        # Add a sample
        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]

        if add_memory:
            self.fifo.add_instance(current_sample)  # for batch-based inference

            with torch.no_grad():
                self.net.eval()
                self.mem.add_instance(current_sample)

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    def tta_attack_mode_turn_on(self):

        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py

                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    def tta_attack_mode_turn_off(self):
        # reset model before actual inference
        self.net = deepcopy(self.original_model)
