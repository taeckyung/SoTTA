import conf
from data_loader.NoisyDataset import NoisyDataset
from data_loader.data_loader import datasets_to_dataloader
from .dnn import DNN
from torch.utils.data import DataLoader, random_split

from utils.loss_functions import *
from utils import memory

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class EATA(DNN):
    def __init__(self, *args, **kwargs):
        super(EATA, self).__init__(*args, **kwargs)

        fisher_dataset = NoisyDataset(base=conf.args.dataset, domains=['test'], transform='val',
                                      noisy_data=conf.args.noisy_type, noisy_data_size=conf.args.noisy_size,
                                      noisy_data_class=conf.args.noisy_class)
        fisher_dataset = \
        random_split(fisher_dataset, [conf.args.fisher_size, len(fisher_dataset) - conf.args.fisher_size])[0]
        fisher_loader = datasets_to_dataloader([fisher_dataset], batch_size=64, concat=True, shuffle=True)

        subnet = configure_model(self.net)
        params, param_names = collect_params(subnet)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets, domains) in enumerate(fisher_loader, start=1):
            if conf.args.gpu_idx is not None:
                images = images.cuda(conf.args.gpu_idx, non_blocking=True)
            # if torch.cuda.is_available():
            #     targets = targets.cuda(conf.args.gpu_idx, non_blocking=True)
            outputs = subnet(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in subnet.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        # logger.info("compute fisher matrices finished")
        del ewc_optimizer

        # self.steps = 1 # SoTTA: replaced to epoch
        # assert self.steps > 0, "EATA requires >= 1 step(s) to forward and update"
        # self.episodic = False  # SoTTA: we don't use episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = conf.args.e_margin  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = conf.args.d_margin  # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers  # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = conf.args.fisher_alpha  # trade-off \beta for two losses (Eqn. 8)

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.net, self.optimizer)

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

        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):
            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)

                if conf.args.tta_attack_type:
                    feats = feats.clone().detach()

                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(feats, self.net,
                                                                                            self.optimizer,
                                                                                            self.fishers,
                                                                                            self.e_margin,
                                                                                            self.current_model_probs,
                                                                                            fisher_alpha=self.fisher_alpha,
                                                                                            num_samples_update=self.num_samples_update_2,
                                                                                            d_margin=self.d_margin)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.net, self.optimizer,
                                      self.model_state, self.optimizer_state)

    def reset_model_probs(self, probs):
        self.current_model_probs = probs


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0,
                           d_margin=0.05, scale_factor=2, num_samples_update=0):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return:
    1. model outputs;
    2. the number of reliable and non-redundant samples;
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model(x)
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]
    # filter redundant samples
    if current_model_probs is not None:
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0),
                                                  outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1]) ** 2).sum()
        loss += ewc_loss
    if x[ids1][ids2].size(0) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with eata."""
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with eata."""
    is_training = model.training
    assert is_training, "eata needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "eata needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "eata should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "eata needs normalization for its optimization"
