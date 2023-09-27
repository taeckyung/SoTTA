from torch.utils.data import DataLoader

import conf
from utils import memory
from utils.loss_functions import *
from utils.sam_optimizer import SAM
from .dnn import DNN

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class SoTTA(DNN):

    def __init__(self, *args, **kwargs):
        super(SoTTA, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False
        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

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

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):  # ablation study
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):  # language models
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)  # required for evaluation
        self.mem_state = self.mem.save_state_dict()

        self.ema = None
        self.batchnorm_stats = []

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

                if conf.args.memory_type in ['FIFO']:
                    self.mem.add_instance(current_sample)

                elif conf.args.memory_type in ['HUS', 'ConfFIFO']:
                    f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                    logit = self.net(f.unsqueeze(0))
                    pseudo_cls = logit.max(1, keepdim=False)[1][0].cpu().numpy()
                    pseudo_conf = F.softmax(logit, dim=1).max(1, keepdim=False)[0][0].cpu().numpy()
                    self.mem.add_instance([f, pseudo_cls, d, pseudo_conf])

                elif conf.args.memory_type in ['CSTU']:
                    f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
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

                else:
                    raise NotImplementedError

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # if not conf.args.use_learned_stats  and evaluation: #batch-based inference
        if evaluation:
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        if conf.args.memory_type in ['CSTU']:
            feats, _ = self.mem.get_memory()
        else:
            feats, _, _ = self.mem.get_memory()

        if len(feats) == 0:
            return TRAINED

        feats = torch.stack(feats)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=False)

        entropy_loss = HLoss(conf.args.temperature)

        for e in range(conf.args.epoch):
            for batch_idx, (feats,) in enumerate(data_loader):
                self.step(loss_fn=entropy_loss, feats=feats)

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    def step(self, loss_fn, feats=None):
        assert (feats is not None)

        if conf.args.tta_attack_type:
            feats = feats.clone().detach()

        self.net.train()
        feats = feats.to(device)
        preds_of_data = self.net(feats)

        loss_first = loss_fn(preds_of_data)

        self.optimizer.zero_grad()

        loss_first.backward()

        if not isinstance(self.optimizer, SAM):
            self.optimizer.step()
        else:
            # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            self.optimizer.first_step(zero_grad=True)

            preds_of_data = self.net(feats)

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second = loss_fn(preds_of_data)

            loss_second.backward()

            self.optimizer.second_step(zero_grad=True)
