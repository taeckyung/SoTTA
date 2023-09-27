import conf
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *
from utils import memory

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class PseudoLabel(DNN):
    def __init__(self, *args, **kwargs):
        super(PseudoLabel, self).__init__(*args, **kwargs)

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

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

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        if evaluation:
            # Evaluate with a batch
            self.evaluation_online(current_num_sample, self.mem.get_memory())

        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)

        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)

                if conf.args.tta_attack_type:
                    feats = feats.clone().detach()

                preds_of_data = self.net(feats)
                pseudo_cls = preds_of_data.max(1, keepdim=False)[1]
                loss = self.class_criterion(preds_of_data, pseudo_cls)

                self.optimizer.zero_grad()
                if conf.args.tta_attack_type:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                self.optimizer.step()

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED
