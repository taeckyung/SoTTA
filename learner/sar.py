from copy import deepcopy

import math

import conf
from utils import memory
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


class SAR(DNN):
    def __init__(self, *args, **kwargs):
        super(SAR, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        self.net.train()

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            # TODO: support use_learned_stats
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.requires_grad_(True)

        # SAR-specific hyperparameters

        if conf.args.noisy_type == "divide":
            num_class = conf.args.opt['num_class'] - len(conf.args.noisy_class)
        else:
            num_class = conf.args.opt['num_class']

        self.margin_e0 = 0.4 * math.log(num_class)  # math.log(1000)
        self.reset_constant_em = 0.2
        self.ema = None

        self.net_state, self.optimizer_state = \
            copy_model_and_optimizer(self.net, self.optimizer)

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)  # required for evaluation
        self.mem_state = self.mem.save_state_dict()

    def reset(self):
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer")
        load_model_and_optimizer(self.net, self.optimizer, self.net_state, self.optimizer_state)
        self.ema = None

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

                if conf.args.memory_type in ['FIFO', 'Reservoir']:
                    self.mem.add_instance(current_sample)

                elif conf.args.memory_type in ['HUS']:
                    f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)

                    logit = self.net(f.unsqueeze(0))
                    psuedo_conf = logit.max(1, keepdim=False)[0][0].cpu()
                    pseudo_cls = logit.max(1, keepdim=False)[1][0]
                    self.mem.add_instance([f, pseudo_cls, d, psuedo_conf])
                else:
                    raise NotImplementedError

        if conf.args.use_learned_stats and evaluation:  # batch-free inference
            self.evaluation_online(current_num_sample,
                                   [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # Evaluate with a batch
        if not conf.args.use_learned_stats and evaluation:  # batch-based inference
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)

        if conf.args.tta_attack_type:
            feats = feats.clone().detach()

        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):
            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)

                self.optimizer.zero_grad()

                preds_of_data = self.net(feats)

                # filtering reliable samples/gradients for further adaptation; first time forward
                entropys = softmax_entropy(preds_of_data)
                filter_ids_1 = torch.where(entropys < self.margin_e0)
                entropys = entropys[filter_ids_1]
                loss = entropys.mean(0)

                if conf.args.tta_attack_type:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
                self.optimizer.first_step(zero_grad=True)

                # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
                entropys2 = softmax_entropy(self.net(feats))
                entropys2 = entropys2[filter_ids_1]
                filter_ids_2 = torch.where(entropys2 < self.margin_e0)
                loss_second = entropys2[filter_ids_2].mean(0)
                if not np.isnan(loss_second.item()):
                    self.ema = update_ema(self.ema, loss_second.item())

                if conf.args.tta_attack_type:
                    loss_second.backward(retain_graph=True)
                else:
                    loss_second.backward()

                self.optimizer.second_step(zero_grad=True)

                if self.ema is not None and self.ema < 0.2:
                    print("ema < 0.2, now reset the model")
                    ema = self.ema
                    self.reset()
                    self.ema = ema

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED
