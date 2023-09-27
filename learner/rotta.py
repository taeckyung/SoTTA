from copy import deepcopy

import torch.optim as optim

import conf
from utils import memory, bn_layers_rotta
from utils.custom_transforms import get_tta_transforms
from utils.loss_functions import *
from .dnn import DNN

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class RoTTA(DNN):
    def __init__(self, *args, **kwargs):
        super(RoTTA, self).__init__(*args, **kwargs)

        self.net.requires_grad_(False)
        self.alpha = 0.05
        normlayer_names = []

        for name, sub_module in self.net.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(self.net, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = bn_layers_rotta.RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = bn_layers_rotta.RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self.alpha)
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.net, name, momentum_bn)

        params, param_names = self.collect_params(self.net)
        self.optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

        net_ema = deepcopy(self.net)
        for param in net_ema.parameters():
            param.detach_()

        self.net_not_ema = self.net
        self.net = net_ema  # set self.net to self.net_ema
        self.net.to(device)

        self.transform = get_tta_transforms(tuple(self.target_train_set[0][0].shape[1:]))
        self.nu = 0.001

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)  # required for evaluation
        self.mem_state = self.mem.save_state_dict()

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

                if conf.args.memory_type in [
                    'CSTU']:  # RoTTA; Category-balanced Sampling with Timeliness and Uncertainty
                    f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                    self.net.eval()
                    self.net_not_ema.eval()
                    ema_out = self.net(f.unsqueeze(0))
                    predict = torch.softmax(ema_out, dim=1)
                    pseudo_label = torch.argmax(predict, dim=1)
                    entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

                    # add into memory
                    for i, data in enumerate(f.unsqueeze(0)):
                        p_l = pseudo_label[i].item()
                        uncertainty = entropy[i].item()
                        current_instance = (data, p_l, uncertainty)
                        self.mem.add_instance(current_instance)
                        # self.current_instance += 1

                        # if self.current_instance % self.update_frequency == 0:
                        #     self.update_model(model, optimizer)
                else:
                    raise NotImplementedError

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # Evaluate with a batch
        if evaluation:  # batch-based inference
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        # setup models
        self.net.train()
        self.net_not_ema.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()
            self.net_not_ema.eval()

        # get memory data
        # sup_data, ages = self.mem.get_memory_rotta_style()
        sup_data, ages = self.mem.get_memory()

        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.net(sup_data)
            stu_sup_out = self.net_not_ema(strong_sup_aug)
            instance_weight = self.timeliness_reweighting(ages)
            l_sup = (softmax_entropy_rotta(stu_sup_out, ema_sup_out) * instance_weight).mean()

        loss = l_sup
        if loss is not None:
            self.optimizer.zero_grad()
            if conf.args.tta_attack_type:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            self.optimizer.step()

        self.net = self.update_ema_variables(self.net, self.net_not_ema, self.nu)

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    def timeliness_reweighting(self, ages):
        if isinstance(ages, list):
            ages = torch.tensor(ages).float().cuda()
        return torch.exp(-ages) / (1 + torch.exp(-ages))

    def update_ema_variables(self, ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def collect_params(self, model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)
