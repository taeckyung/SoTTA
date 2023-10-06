import random
import torch.utils.data
import torchvision.transforms as transforms

import time
import numpy as np
from torchvision.datasets import ImageFolder

import conf

from data_loader.CIFAR100Dataset import CIFAR100Dataset
from data_loader.CIFAR10Dataset import CIFAR10Dataset
from data_loader.IMAGENETDataset import ImageNetDataset
from utils.normalize_layer import *

opt10 = conf.CIFAR10Opt
opt100 = conf.CIFAR100Opt
imagenet = conf.IMAGENET_C
mnist = conf.MNISTOpt

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(conf.args.gpu_idx)  #

NOISY_CLASS_IDX = 10000


class NoisyDataset(torch.utils.data.Dataset):
    """
    Noisy data stream
    Currently supports cifar10noisy, imagenetnoisy, and cifar100noisy
    with noise data of original, cifar100, imagenet, mnist, uniform, and repeat.
    """
    noisy_data_types = ["original", "divide", "repeat", "oneclassrepeat", "cifar100", "cifar100c",
                        "gaussian", "uniform", "mnist", "cifar10", "imagenet"]

    def __init__(self, base="cifar10noisy",
                 domains=None, activities=None, transform='none',
                 noisy_data=None, noisy_data_size=None, noisy_data_class=None):
        self.domains = domains
        self.activity = activities
        self.noisy_data = noisy_data
        self.noisy_data_size = noisy_data_size
        self.noisy_data_class = noisy_data_class

        self.domain = domains[0]

        self.img_shape = opt10['img_size']
        self.features = None
        self.class_labels = None
        self.domain_labels = None

        self.file_path10 = opt10['file_path']
        self.file_path100 = opt100['file_path']
        self.file_path_imagenet = imagenet['file_path']
        self.file_path_mnist = mnist['file_path']

        assert (base in ["cifar10noisy", "imagenetnoisy", "cifar100noisy"])
        assert (self.noisy_data in self.noisy_data_types)
        assert not (self.noisy_data in ['oneclassrepeat', 'gaussian', 'uniform'] and self.noisy_data_size is None)
        assert not (self.noisy_data in ['divide', 'oneclassrepeat'] and self.noisy_data_class is None)

        if transform == 'src':
            self.is_src = True
        else:
            self.is_src = False

        if base == 'cifar10noisy':
            if self.domain == 'none':
                self.base_dataset = CIFAR10Dataset(domains=['original'], transform=transform)
            else:
                self.base_dataset = CIFAR10Dataset(domains=self.domains, transform=transform)
            self.img_size = 32
            if transform == 'src':
                self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                    ])

            elif transform == 'val':
                self.transform = None
            else:
                raise NotImplementedError
        elif base == 'cifar100noisy':
            if self.domain == 'none':
                self.base_dataset = CIFAR100Dataset(domains=['original'], transform=transform)
            else:
                self.base_dataset = CIFAR100Dataset(domains=self.domains, transform=transform)
            self.img_size = 32
            if transform == 'src':
                self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                    ])

            elif transform == 'val':
                self.transform = None
            else:
                raise NotImplementedError
        elif base == 'imagenetnoisy':
            self.base_dataset = ImageNetDataset(domain=self.domain, transform=transform)
            self.img_size = 224

            if transform == 'src':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            elif transform == 'val':
                if self.domain.startswith('original') or self.domain.startswith('test'):
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])

                if self.noisy_data in ["oneclassrepeat", "divide", "repeat"]:
                    self.base_dataset.load_features()  # imagenet does not originally load features
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        self.preprocessing()

    def resample(self, data: np.ndarray, size: int):
        """
        Resample the input data to corresponding size.
        """
        if size == len(data):
            return data
        elif size < len(data):
            return data[random.sample(range(len(data)), size)]
        else:
            remainder = data[random.sample(range(len(data)), size % len(data))]
            repeat = np.tile(data, (int(size / len(data)), 1, 1, 1))
            return np.concatenate((repeat, remainder), axis=0)

    def preprocessing(self):
        if self.noisy_data == "cifar100":  # cifar100
            path100 = f'{self.file_path100}/origin/'
            data100 = np.load(path100 + 'original.npy')
            # change NHWC to NCHW format
            data100 = np.transpose(data100, (0, 3, 1, 2))
            # make it compatible with our models (normalize)
            data100 = data100.astype(np.float32) / 255.0

            if self.noisy_data_size:
                data100 = self.resample(data100, self.noisy_data_size)

            tr = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            data100 = np.array([tr(torch.Tensor(img)).numpy() for img in data100])

            noisy_stream = data100

        elif self.noisy_data == "cifar100c":  # cifar100c
            path100c = f'{self.file_path100}/corrupted/severity-{self.domains[0][-1]}/'
            data_filename = self.domains[0].split('-')[0] + '.npy'
            data100c = np.load(path100c + data_filename)
            # change NHWC to NCHW format
            data100c = np.transpose(data100c, (0, 3, 1, 2))
            # make it compatible with our models (normalize)
            data100c = data100c.astype(np.float32) / 255.0

            tr = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            data100c = np.array([tr(torch.Tensor(img)).numpy() for img in data100c])

            noisy_stream = data100c

        elif self.noisy_data == "cifar10":  # cifar10
            path10 = f'{self.file_path10}/origin/'
            data10 = np.load(path10 + 'original.npy')
            # change NHWC to NCHW format
            data10 = np.transpose(data10, (0, 3, 1, 2))
            # make it compatible with our models (normalize)
            data10 = data10.astype(np.float32) / 255.0

            if self.noisy_data_size:
                data10 = data10[random.sample(range(len(data10)), self.noisy_data_size)]

            tr = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            data10 = np.array([tr(torch.Tensor(img)).numpy() for img in data10])

            noisy_stream = data10

        elif self.noisy_data == "imagenet":  # imagenet validation set
            path = f'{self.file_path_imagenet}/origin/Data/CLS-LOC/val/'

            tr = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])

            features = []

            dataset = ImageFolder(path, transform=tr)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=False,
                                                     drop_last=False)
            
            for b_i, data in enumerate(dataloader):  # must be loaded from dataloader, due to transform in the __getitem__()
                feat, _ = data
                # convert a batch of tensors to list, and then append to our list one by one
                feats = torch.unbind(feat, dim=0)
                for i in range(len(feats)):
                    features.append(feats[i])

            noisy_stream = np.stack(features)
            if self.noisy_data_size:
                noisy_stream = noisy_stream[random.sample(range(len(noisy_stream)), self.noisy_data_size)]

        elif self.noisy_data == "gaussian":  # gaussian noise
            gaussian_img = np.random.normal(0, 1, size=(self.noisy_data_size, 3, self.img_size, self.img_size))
            norm_gaussian_img = np.float32((gaussian_img - np.min(gaussian_img)) / np.ptp(gaussian_img))
            noisy_stream = norm_gaussian_img

        elif self.noisy_data == "repeat":  # repeat base dataset
            noisy_stream = self.base_dataset.features

        elif self.noisy_data == "uniform":  # uniform50
            noisy_stream = np.float32(np.random.random(size=(self.noisy_data_size, 3, self.img_size, self.img_size)))

        elif self.noisy_data == "mnist":  # mnist test set
            path_mnist = f'{self.file_path_mnist}/identity/'
            data_mnist = np.load(path_mnist + 'test_images.npy')

            # change NHWC to NCHW format
            data_mnist = np.transpose(data_mnist, (0, 3, 1, 2))
            # make it compatible with our models (normalize)
            data_mnist = data_mnist.astype(np.float32) / 255.0

            if self.noisy_data_size:
                data_mnist = self.resample(data_mnist, self.noisy_data_size)

            mnist_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(3),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            data_mnist = np.array([mnist_transform(torch.Tensor(img)).numpy() for img in data_mnist])

            noisy_stream = data_mnist

        elif self.noisy_data == "original":
            noisy_stream = np.array([])
        else:
            raise NotImplementedError

        if self.noisy_data == "original" or self.is_src is True:
            self.dataset = self.base_dataset.dataset
        else:
            noisy_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(noisy_stream),  
                torch.from_numpy(NOISY_CLASS_IDX * np.ones((len(noisy_stream)))),
                torch.from_numpy(np.array(np.zeros((len(noisy_stream))))))

            if self.domain == "none":
                self.dataset = noisy_dataset
            else:
                self.dataset = torch.utils.data.ConcatDataset([noisy_dataset, self.base_dataset.dataset])

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        data = self.dataset[idx]
        if len(data) == 3:
            img, cl, dl = data
        elif len(data) == 2:
            img, cl, dl = data[0], torch.tensor(data[1]), torch.tensor(0)  # avoid errors on torch.stack()
        else:
            raise NotImplementedError

        if int(cl) != int(NOISY_CLASS_IDX):  # in-dist data
            if self.transform:
                img = self.transform(img)

        return img, cl, dl


if __name__ == '__main__':
    pass
