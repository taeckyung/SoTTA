import random

import numpy as np
import torch

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator


class FIFO:
    def __init__(self, capacity):
        self.data = [[], [], []]
        self.capacity = capacity
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 3)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass


class HUS:
    def __init__(self, capacity, threshold=None):
        self.data = [[[], [], [], []] for _ in
                     range(conf.args.opt['num_class'])]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        self.threshold = threshold

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold

        return dic

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 4)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_average_confidence(self):
        conf_list = []
        for i, data_per_cls in enumerate(self.data):
            for confidence in data_per_cls[3]:
                conf_list.append(confidence)
        if len(conf_list) > 0:
            return np.average(conf_list)
        else:
            return 0

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][3])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][3])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], [], []] for _ in range(conf.args.opt['num_class'])]  # feat, pseudo_cls, domain, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][2].append(0)
            self.data[tgt_idx][3].append(aux[i])


class ConfFIFO:
    def __init__(self, capacity, threshold):
        self.data = [[], [], [], []]
        self.capacity = capacity
        self.threshold = threshold
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        self.threshold = state_dict['threshold']
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold
        return dic

    def get_memory(self):
        return self.data[:3]

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 4)

        if instance[3] < self.threshold:
            return

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

    def reset_value(self, feats, cls, aux):
        self.data = [[], [], [], []]
