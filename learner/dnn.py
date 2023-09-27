import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import conf
import torch.nn.functional as F

from data_loader.NoisyDataset import NOISY_CLASS_IDX
from data_loader.data_loader import load_cache, save_cache
from utils import memory

from utils.logging import *
from utils.loss_functions import softmax_entropy, calc_energy
from utils.normalize_layer import *
from utils.sam_optimizer import SAM, sam_collect_params
from utils import memory_rotta
from copy import deepcopy

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator


class DNN:
    def __init__(self, model, source_dataloader, target_dataloader, write_path):
        self.device = device

        # init dataloader
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader

        if conf.args.dataset in ['cifar10', 'cifar100'] and conf.args.tgt_train_dist == 0:
            self.tgt_train_dist = 4  # Dirichlet is default for non-real-distribution data
        else:
            self.tgt_train_dist = conf.args.tgt_train_dist

        self.target_data_cache()

        self.write_path = write_path

        ################## Init & prepare model###################
        self.conf_list = []

        # Load model
        if 'resnet' in conf.args.model:
            if conf.args.dataset not in ['imagenet', 'imagenetnoisy']:
                num_feats = model.fc.in_features
                if conf.args.noisy_type == "divide":
                    num_class = conf.args.opt['num_class'] - len(conf.args.noisy_class)
                else:
                    num_class = conf.args.opt['num_class']
                model.fc = nn.Linear(num_feats, num_class)  # match class number
            self.net = model
        else:
            self.net = model.Net()

        if conf.args.load_checkpoint_path:  # false if conf.args.load_checkpoint_path==''
            self.load_checkpoint(conf.args.load_checkpoint_path)

        # Add normalization layers
        norm_layer = get_normalize_layer(conf.args.dataset)
        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)

        if conf.args.parallel and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        self.net.to(device)

        ##########################################################

        # init criterions, optimizers, scheduler
        if conf.args.method == 'Src':
            if conf.args.dataset in ['cifar10', 'cifar100', 'cifar10noisy', 'cifar100noisy']:
                self.optimizer = torch.optim.SGD(
                    self.net.parameters(),
                    conf.args.opt['learning_rate'],
                    momentum=conf.args.opt['momentum'],
                    weight_decay=conf.args.opt['weight_decay'],
                    nesterov=True)

                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=conf.args.epoch * len(
                    self.source_dataloader['train']))
            elif conf.args.dataset in ['tinyimagenet', 'imagenet', 'imagenetnoisy']:
                self.optimizer = torch.optim.SGD(
                    self.net.parameters(),
                    conf.args.opt['learning_rate'],
                    momentum=conf.args.opt['momentum'],
                    weight_decay=conf.args.opt['weight_decay'],
                    nesterov=True)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                            weight_decay=conf.args.opt['weight_decay'])
        elif conf.args.method == 'MEMO':
            if conf.args.dataset in ['cifar10', 'cifar100', 'cifar10noisy', 'imagenetnoisy', 'cifar100noisy']:
                self.optimizer = torch.optim.SGD(
                    self.net.parameters(),
                    conf.args.opt['learning_rate'])
            else:
                raise NotImplementedError

        elif conf.args.method == 'SAR':
            # TODO: set base optimizer dynamically (current: SGD)
            params, _ = sam_collect_params(self.net, freeze_top=True)
            self.optimizer = SAM(params, torch.optim.SGD, lr=conf.args.opt['learning_rate'],
                                 momentum=conf.args.opt['momentum'])
        elif conf.args.method in ['SoTTA', 'LOG'] and conf.args.esm:
            params, _ = sam_collect_params(self.net, freeze_top=True)
            self.optimizer = SAM(params, torch.optim.Adam, rho=0.05, lr=conf.args.opt['learning_rate'],
                                 weight_decay=conf.args.opt['weight_decay'])
        elif conf.args.method == "RoTTA":
            # self.optimizer = optim.Adam(self.net.parameters(),lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
            self.optimizer = None
        elif conf.args.method == "EATA":
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=conf.args.opt['learning_rate'],
                                       momentum=conf.args.opt['momentum'],
                                       weight_decay=conf.args.opt['weight_decay'])
        else:
            if conf.args.method == 'TENT' and conf.args.dataset == 'imagenetnoisy':  # TENT use SGD for imagenet
                self.optimizer = optim.SGD(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                           weight_decay=conf.args.opt['weight_decay'])
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                            weight_decay=conf.args.opt['weight_decay'])

        self.class_criterion = nn.CrossEntropyLoss()
        self.attack_loss = nn.CrossEntropyLoss()

        # online learning
        if conf.args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'HUS':
            self.mem = memory.HUS(capacity=conf.args.memory_size, threshold=conf.args.high_threshold)
        elif conf.args.memory_type == 'CSTU':
            self.mem = memory_rotta.CSTU(capacity=conf.args.memory_size, num_class=conf.args.opt['num_class'],
                                         lambda_t=1, lambda_u=1)  # replace memory with original RoTTA
        elif conf.args.memory_type == 'ConfFIFO':
            self.mem = memory.ConfFIFO(capacity=conf.args.memory_size, threshold=conf.args.high_threshold)

        self.json = {}
        self.l2_distance = []
        self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

        if conf.args.tta_attack_type:
            self.purturb = None
            self.index_attack = None

        self.fifo = None

    def target_data_cache(self):
        dataset = conf.args.dataset
        cond = conf.args.tgt

        filename = f"{dataset}_{conf.args.noisy_type}_{conf.args.noisy_size}_{conf.args.noisy_class}_" \
                   f"{conf.args.seed}_{conf.args.tgt_train_dist}"

        file_path = conf.args.opt['file_path'] + "_target_train_set"

        self.target_train_set = load_cache(filename, cond, file_path, transform=None)

        if not self.target_train_set:
            self.target_data_processing()
            save_cache(self.target_train_set, filename, cond, file_path, transform=None)

    def target_data_processing(self):

        features = []
        cl_labels = []
        do_labels = []

        for b_i, (feat, cl, dl) in enumerate(self.target_dataloader['train']):
            # must be loaded from dataloader, due to transform in the __getitem__()
            features.append(feat.squeeze(0))
            cl_labels.append(cl.squeeze())
            do_labels.append(dl.squeeze())

        tmp = list(zip(features, cl_labels, do_labels))

        features, cl_labels, do_labels = zip(*tmp)
        features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

        num_class = conf.args.opt['num_class']

        result_feats = []
        result_cl_labels = []
        result_do_labels = []

        # real distribution
        if self.tgt_train_dist == 0:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # random distribution
        elif self.tgt_train_dist == 1:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = np.random.randint(len(features))
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # dirichlet distribution
        elif self.tgt_train_dist == 4:
            dirichlet_numchunks = conf.args.opt['num_class']

            tgt_class_list = list(range(num_class))
            if conf.args.noisy_type:
                tgt_class_list += [NOISY_CLASS_IDX]

            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
            min_size = -1
            N = len(features)
            min_size_thresh = 10  # if conf.args.dataset in ['tinyimagenet'] else 10
            while min_size < min_size_thresh:  # prevent any chunk having too less data
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]  # contains data per each class
                for k in tgt_class_list:
                    cl_labels_np = torch.Tensor(cl_labels).numpy()
                    idx_k = np.where(cl_labels_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(conf.args.dirichlet_beta, dirichlet_numchunks))

                    # balance
                    proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                            zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    # store class-wise data
                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []

            # create temporally correlated toy dataset by shuffling classes
            for chunk in idx_batch_cls:
                cls_seq = list(range(len(tgt_class_list)))
                np.random.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    result_feats.extend([features[i] for i in idx])
                    result_cl_labels.extend([cl_labels[i] for i in idx])
                    result_do_labels.extend([do_labels[i] for i in idx])
                    sequence_stats.extend(list(np.repeat(cls, len(idx))))

            # trim data if num_sample is smaller than the original data size
            num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
            result_feats = result_feats[:num_samples]
            result_cl_labels = result_cl_labels[:num_samples]
            result_do_labels = result_do_labels[:num_samples]

        elif self.tgt_train_dist == 5:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        elif self.tgt_train_dist == 6:  # for SAR online imbalanced label shifts
            assert conf.args.dataset in ['imagenet']

            ir = 500000  # just use fixed `ir` for now. Depends on the dataset.

            if conf.args.seed == 2021:
                indices_path = './data_loader/imagenetc_imbalanced/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(
                    100000, ir)
            else:
                indices_path = './data_loader/imagenetc_imbalanced/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(
                    conf.args.seed, 100000, ir)

            indices = np.load(indices_path).astype(int).tolist()
            result_feats = [features[i] for i in indices]
            result_cl_labels = [cl_labels[i] for i in indices]
            result_do_labels = [do_labels[i] for i in indices]

        remainder = len(result_feats) % conf.args.update_every_x  # drop leftover samples
        if remainder == 0:
            pass
        else:
            result_feats = result_feats[:-remainder]
            result_cl_labels = result_cl_labels[:-remainder]
            result_do_labels = result_do_labels[:-remainder]

        try:
            self.target_train_set = (torch.stack(result_feats),
                                     torch.stack(result_cl_labels),
                                     torch.stack(result_do_labels))
        except:
            try:
                self.target_train_set = (torch.stack(result_feats),
                                         result_cl_labels,
                                         torch.stack(result_do_labels))
            except:  # for dataset which each image has different shape
                self.target_train_set = (result_feats,
                                         result_cl_labels,
                                         torch.stack(result_do_labels))

    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0], NormalizeLayer):
                cp = self.net[1]
        else:
            cp = self.net

        if isinstance(self.net, nn.DataParallel):
            cp = self.net.module

        torch.save(cp.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
        self.net.load_state_dict(self.checkpoint, strict=True)
        self.net.to(device)

    def get_loss_and_confusion_matrix(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)

        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm, preds_of_data

    def get_loss_cm_error(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)
        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        assert (len(label) == len(pred_label))
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        errors = [0 if label[i] == pred_label[i] else 1 for i in range(len(label))]
        return loss_of_data, cm, errors

    def log_loss_results(self, condition, epoch, loss_avg):

        if condition == 'train_online':
            # print loss
            print('{:s}: [current_sample: {:d}]'.format(
                condition, epoch
            ))
        else:
            # print loss
            print('{:s}: [epoch: {:d}]\tLoss: {:.6f} \t'.format(
                condition, epoch, loss_avg
            ))

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        assert (condition in ['valid', 'test'])
        # assert (suffix in ['labeled', 'unlabeled', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))

        return class_accuracy

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.net.train()

        class_loss_sum = 0.0

        total_iter = 0

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter

            for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                feats, cls, _ = labeled_data
                feats, cls = feats.to(device), cls.to(device)

                if torch.isnan(feats).any() or torch.isinf(feats).any():  # For reallifehar debugging
                    print('invalid input detected at iteration ', batch_idx)
                    exit(1)
                # compute the feature
                preds = self.net(feats)
                if torch.isnan(preds).any() or torch.isinf(preds).any():  # For reallifehar debugging
                    print('invalid input detected at iteration ', batch_idx)
                    exit(1)
                class_loss = self.class_criterion(preds, cls)
                class_loss_sum += float(class_loss * feats.size(0))

                if torch.isnan(class_loss).any() or torch.isinf(class_loss).any():  # For reallifehar debugging
                    print('invalid input detected at iteration ', batch_idx)
                    exit(1)

                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()
                if conf.args.dataset in ['cifar10', 'cifar100', 'cifar10noisy', 'cifar100noisy']:
                    self.scheduler.step()

        ######################## LOGGING #######################

        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')

    def evaluation(self, epoch, condition):
        # Evaluate with a batch of samples, which is a typical way of evaluation. Used for pre-training or offline eval.

        self.net.eval()

        with torch.no_grad():
            inputs, cls, dls = self.target_train_set
            tgt_inputs = inputs.to(device)
            tgt_cls = cls.to(device)

            preds = self.net(tgt_inputs)

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(preds, tgt_cls)
            y_pred = preds.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(tgt_cls.cpu(), y_pred.cpu(), labels=labels)

        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, class_loss_of_test_data
        ))
        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))

        self.logger('accuracy', class_accuracy, epoch, condition)
        self.logger('loss', class_loss_of_test_data, epoch, condition)

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def copy_model_and_optimizer(self, model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    def load_model_and_optimizer(self, model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    def evaluation_online_body(self, epoch, current_samples, feats, cls, dls):

        if conf.args.log_grad:
            model_state, optimizer_state = self.copy_model_and_optimizer(self.net, self.optimizer)

        cls = cls.to(torch.int32)  # bugfix when comparing noisy_index
        feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)
        grad_norms = 0.0

        if conf.args.method == 'LAME':
            y_pred = self.batch_evaluation(feats)  # already softmax
            y_conf = y_pred.max(-1)[0].cpu().numpy()
            y_pred = y_pred.argmax(-1).cpu().numpy()

            entropy = np.zeros_like(y_pred)  # we don't support this now
            y_energy = np.zeros_like(entropy)  # we don't support this now

        elif conf.args.method == 'CoTTA':
            x = feats
            anchor_prob = torch.nn.functional.softmax(self.net_anchor(x), dim=1).max(1)[0]
            standard_ema = self.net_ema(x)

            N = 32
            outputs_emas = []

            # Threshold choice discussed in supplementary
            # enable data augmentation for vision datasets
            if anchor_prob.mean(0) < self.ap:
                for i in range(N):
                    outputs_ = self.net_ema(self.transform(x)).detach()
                    outputs_emas.append(outputs_)
                outputs_ema = torch.stack(outputs_emas).mean(0)
            else:
                outputs_ema = standard_ema
            y_pred = outputs_ema
            entropy = softmax_entropy(y_pred)
            y_conf = F.softmax(y_pred, dim=1).max(1, keepdim=False)[0]
            y_energy = calc_energy(y_pred).cpu()
            y_pred = y_pred.max(1, keepdim=False)[1]

        elif conf.args.method == 'TPT':
            with torch.cuda.amp.autocast():
                y_pred = self.net(feats)
                entropy = softmax_entropy(y_pred)
                y_conf = F.softmax(y_pred, dim=1).max(1, keepdim=False)[0]
                y_energy = calc_energy(y_pred).cpu()
                y_pred = y_pred.max(1, keepdim=False)[1]

        else:
            y_pred = self.net(feats)
            entropy = softmax_entropy(y_pred)
            y_conf = F.softmax(y_pred, dim=1).max(1, keepdim=False)[0]
            y_energy = calc_energy(y_pred).cpu()
            y_pred = y_pred.max(1, keepdim=False)[1]

            if conf.args.log_grad:
                if isinstance(self.optimizer, SAM):
                    self.optimizer.zero_grad()
                    # filtering reliable samples/gradients for further adaptation; first time forward
                    loss = entropy.mean(0)
                    loss.backward()

                    # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
                    self.optimizer.first_step(zero_grad=True)

                    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
                    entropys2 = softmax_entropy(self.net(feats))
                    loss_second = entropys2.mean(0)
                    loss_second.backward()

                    grad_norms = self.optimizer._grad_norm().item()
                else:
                    self.optimizer.zero_grad()
                    loss = entropy.mean(0)
                    loss.backward()
                    grad_norms = torch.norm(
                        torch.stack([
                            (1.0 * p.grad).norm(p=2)
                            for p in self.net.parameters()
                            if p.grad is not None
                        ]),
                        p=2
                    ).item()

        ###################### SAVE RESULT
        # get lists from json

        try:
            true_cls_list = self.json['gt']
            pred_cls_list = self.json['pred']
            accuracy_list = self.json['accuracy']
            f1_macro_list = self.json['f1_macro']
            distance_l2_list = self.json['distance_l2']
            noisy_pred_cls_list = self.json['pred_noisy']
            conf_list = self.json['confidence']
            noisy_conf_list = self.json['confidence_noisy']
            entropy_list = self.json['entropy']
            noisy_entropy_list = self.json['entropy_noisy']
            ood_pred = self.json['ood_pred']
            ood_gt = self.json['ood_gt']
            gt_noisy = self.json['gt_noisy']
            energy_list = self.json['energy']
            noisy_energy_list = self.json['energy_noisy']
            grad_list = self.json['grad']
            noisy_grad_list = self.json['grad_noisy']
            total_pred_list = self.json['pred_total']
            total_conf_list = self.json['confidence_total']
            total_entropy_list = self.json['entropy_total']
            total_grad_list = self.json['grad_total']
        except KeyError:
            true_cls_list = []
            pred_cls_list = []
            accuracy_list = []
            f1_macro_list = []
            distance_l2_list = []
            noisy_pred_cls_list = []
            conf_list = []
            noisy_conf_list = []
            entropy_list = []
            noisy_entropy_list = []
            ood_pred = []
            ood_gt = []
            gt_noisy = []
            energy_list = []
            noisy_energy_list = []
            grad_list = []
            noisy_grad_list = []
            total_pred_list = []
            total_conf_list = []
            total_entropy_list = []
            total_grad_list = []

        if conf.args.noisy_type:
            cls = cls.cpu().to(torch.int32)
            noisy_y_pred = y_pred[cls == NOISY_CLASS_IDX]
            y_pred = y_pred[cls != NOISY_CLASS_IDX]

            noisy_pred_cls_list += [int(c) for c in noisy_y_pred.tolist()]
            noisy_conf_list += [float(c) for c in y_conf[cls == NOISY_CLASS_IDX].tolist()]
            noisy_entropy_list += [float(c) for c in entropy[cls == NOISY_CLASS_IDX].tolist()]
            noisy_energy_list += [float(c) for c in y_energy[cls == NOISY_CLASS_IDX].tolist()]
            if conf.args.log_grad:
                if cls[0] != NOISY_CLASS_IDX:
                    grad_list += [float(grad_norms)]
                else:
                    noisy_grad_list += [float(grad_norms)]

        # append values to lists
        true_cls_list += [int(c) for c in cls if c != NOISY_CLASS_IDX]
        pred_cls_list += [int(c) for c in y_pred.tolist() if c != NOISY_CLASS_IDX]
        conf_list += [float(c) for c in y_conf[cls != NOISY_CLASS_IDX].tolist()]
        entropy_list += [float(c) for c in entropy[cls != NOISY_CLASS_IDX].tolist()]
        energy_list += [float(c) for c in y_energy[cls != NOISY_CLASS_IDX].tolist()]

        total_pred_list += [int(c) for c in y_pred.tolist()]
        total_conf_list += [float(c) for c in y_conf.tolist()]
        total_entropy_list += [float(c) for c in entropy.tolist()]
        total_grad_list += [float(grad_norms)]

        if len(true_cls_list) > 0:
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) / float(
                len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)
            f1_macro_list.append(f1_score(true_cls_list, pred_cls_list,
                                          average='macro'))

            self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

            # epoch: 1~len(self.target_train_set[0])
            progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):  # consider a batch input
                if i in progress_checkpoint:
                    print(
                        f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

        # update self.json file
        self.json = {
            'gt': true_cls_list,
            'pred': pred_cls_list,
            'accuracy': accuracy_list,
            'f1_macro': f1_macro_list,
            'distance_l2': distance_l2_list,
            'pred_noisy': noisy_pred_cls_list,
            'confidence': conf_list,
            'confidence_noisy': noisy_conf_list,
            'entropy': entropy_list,
            'entropy_noisy': noisy_entropy_list,
            'ood_pred': ood_pred,
            'ood_gt': ood_gt,
            'gt_noisy': gt_noisy,
            'energy': energy_list,
            'energy_noisy': noisy_energy_list,
            'grad': grad_list,
            'grad_noisy': noisy_grad_list,
            'pred_total': total_pred_list,
            'confidence_total': total_conf_list,
            'entropy_total': total_entropy_list,
            'grad_total': total_grad_list
        }

        if conf.args.log_grad:
            self.load_model_and_optimizer(self.net, self.optimizer, model_state, optimizer_state)

    def evaluation_online(self, epoch, current_samples):
        # Evaluate with online samples that come one by one while keeping the order.
        self.net.eval()

        if conf.args.log_grad:  # run for each sample to log per-sample gradient
            for feature, cl_label, do_label in zip(*current_samples):
                feats, cls, dls = (torch.stack([feature]), torch.stack([cl_label]), torch.stack([do_label]))
                self.evaluation_online_body(epoch, current_samples, feats, cls, dls)
        else:  # batch-based
            with torch.no_grad():  # we don't log grad here
                # extract each from list of current_sample
                features, cl_labels, do_labels = current_samples
                feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
                self.evaluation_online_body(epoch, current_samples, feats, cls, dls)

    def dump_eval_online_result(self, is_train_offline=False):

        if is_train_offline:

            feats, cls, dls = self.target_train_set

            batchsize = conf.args.opt['batch_size']

            for num_sample in range(0, len(feats), batchsize):
                current_sample = feats[num_sample:num_sample + batchsize], cls[num_sample:num_sample + batchsize], dls[
                                                                                                                   num_sample:num_sample + batchsize]
                self.evaluation_online(num_sample + batchsize,
                                       [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])

        # logging json files
        json_file = open(self.write_path + 'online_eval.json', 'w')
        json_subsample = {key: self.json[key] for key in self.json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """

        #### for test data
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss

    def evaluation_online_num_sample(self, current_num_sample):
        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        self.evaluation_online(current_num_sample, [[current_sample[0]], [current_sample[1]], [current_sample[2]]])
        self.log_loss_results('detect ood : skip tta', epoch=current_num_sample, loss_avg=0)
        return SKIPPED

    # implementation of the paper on 3 attack scenarios on TTA
    # implement version assume that params_{t} are almost similar with params_{t+1}
    def reset_purturb(self):
        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        # get data from evaluation part
        if self.fifo:
            feats, cls, dls = self.fifo.get_memory()
        else:
            raise NotImplementedError

        if conf.args.noisy_type == "repeat":
            self.attack_index = []
            self.benign_index = []
            for i in range(len(cls)):
                if int(cls[i]) == int(NOISY_CLASS_IDX):
                    self.attack_index.append(i)
                else:
                    self.benign_index.append(i)
            num_noisy = len(self.attack_index)

        else:
            num_noisy = conf.args.tta_attack_num_samples
            shuffled_index = torch.randperm(len(cls))
            attack_index = torch.sort(shuffled_index[:num_noisy]).values
            benign_index = torch.sort(shuffled_index[num_noisy:]).values
            self.attack_index = attack_index.tolist()
            self.benign_index = benign_index.tolist()

        self.benign_cls = torch.tensor([cls[i] for i in self.benign_index]).long().to(device)

        from torch.autograd import Variable

        self.purturb = []
        for _ in range(num_noisy):
            currnet_purturb = Variable(torch.zeros(feats[0].shape))
            currnet_purturb.requires_grad = True
            currnet_purturb = currnet_purturb.to(device)
            self.purturb.append(currnet_purturb)

        return TRAINED

    def tta_attack_train(self, current_num_sample, step):

        if len(self.purturb) == 0:  # No noisy samples in the batch
            return

        if self.fifo:
            feats, cls, dls = self.fifo.get_memory()
        else:
            raise NotImplementedError

        ls_feats_updated = []
        index_puturb = 0

        self.mem.set_memory(self.mem_state)  # reset mem for adding new generated attack samples

        for i in range(len(cls)):

            if i in self.attack_index:
                ls_feats_updated.append(torch.clamp(feats[i].to(device) + self.purturb[index_puturb], min=0, max=1))
                index_puturb += 1
            else:
                ls_feats_updated.append(feats[i].to(device))

            current_sample = (ls_feats_updated[-1], cls[i], dls[i])
            # add attacked samples in mem for adaptation
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
                    # RoTTA; Category-balanced Sampling with Timeliness and Uncertainty
                    with torch.no_grad():
                        f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(
                            device)
                        self.net.eval()
                        if conf.args.method in ['RoTTA']:
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

                else:
                    raise NotImplementedError

        self.train_online(current_num_sample, add_memory=False, evaluation=False)

        feats_updated = torch.stack(ls_feats_updated).to(device)

        if conf.args.method == "SRC_FOR_TTA_ATTACK":
            self.tta_attack_mode_turn_on()

        self.net.eval()

        preds = self.net(feats_updated)

        # loss calculation
        if conf.args.tta_attack_type == "targeted":
            raise NotImplementedError

        elif conf.args.tta_attack_type == "indiscriminate":
            loss = -self.attack_loss(preds[self.benign_index], self.benign_cls)

            acc = torch.sum(torch.argmax(preds[self.benign_index], axis=1) == self.benign_cls) / len(self.benign_cls)
            print('{:s}: [epoch: {:d}]\tLoss: {:.6f} \tAcc: {:.6f}'.format("tta_attack_train", step, loss, acc))

        elif conf.args.tta_attack_type == "stealthy":
            raise NotImplementedError

        else:
            raise NotImplementedError

        # update perturb
        alpha = 1.0 / 255

        grad = torch.autograd.grad(loss, self.purturb, retain_graph=False, create_graph=False)

        for i in range(len(self.purturb)):
            self.purturb[i] -= alpha * torch.sign(grad[i])
            self.purturb[i] = torch.clamp(self.purturb[i], min=-conf.args.tta_attack_eps, max=conf.args.tta_attack_eps)

    def tta_attack_update(self, current_num_sample):  # update memory with final perturbation
        # get data from evaluation part
        if self.fifo:
            feats, cls, dls = self.fifo.get_memory()
        else:
            raise NotImplementedError

        feats_updated = []
        index_puturb = 0

        self.mem.set_memory(self.mem_state)  # reset mem for adding new generated attack samples

        for i in range(len(cls)):

            if i in self.attack_index:
                feats_updated.append(torch.clamp(feats[i] + self.purturb[index_puturb].detach().cpu(), min=0, max=1))
                index_puturb += 1
            else:
                feats_updated.append(feats[i])

            # add attacked samples in mem for adaptation
            current_sample = (feats_updated[-1], cls[i], dls[i])
            with torch.no_grad():
                self.net.eval()

                if conf.args.memory_type in ['FIFO', 'Reservoir']:
                    self.mem.add_instance(current_sample)

                elif conf.args.memory_type in ['HUS', 'ConfFIFO']:
                    f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                    logit = self.net(f.unsqueeze(0))
                    pseudo_cls = logit.max(1, keepdim=False)[1][0].cpu().numpy()
                    pseudo_conf = F.softmax(logit, dim=1).max(1, keepdim=False)[0][0].cpu().numpy()
                    self.mem.add_instance([f, pseudo_cls, d, pseudo_conf])

                elif conf.args.memory_type in [
                    'CSTU']:  # RoTTA; Category-balanced Sampling with Timeliness and Uncertainty
                    with torch.no_grad():
                        f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(
                            device)
                        self.net.eval()
                        if conf.args.method in ['RoTTA']:
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

                else:
                    raise NotImplementedError

        # update fifo memory
        dic = {'data': [feats_updated, cls, dls]}
        self.fifo.set_memory(dic)
        self.mem_state = self.mem.save_state_dict()  # update final mem_state

        if conf.args.method == "SRC_FOR_TTA_ATTACK":
            self.tta_attack_mode_turn_off()
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        else:
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        return
