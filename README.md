# SoTTA: Robust Test-Time Adaptation on Noisy Data Streams (NeurIPS '23)

This is the official PyTorch Implementation of 
"SoTTA: Robust Test-Time Adaptation on Noisy Data Streams (NeurIPS '23)" by 
[Taesik Gong*](https://taesikgong.com/), 
[Yewon Kim*](https://yewon-kim.com/), 
[Taeckyung Lee*](https://taeckyung.github.io/), 
Sorn Chottananurak, and 
[Sung-Ju Lee](https://sites.google.com/site/wewantsj/) (* Equal contribution).

[[ OpenReview ]](https://openreview.net/forum?id=3bdXag2rUd) [[ arXiv ]](https://arxiv.org/abs/2310.10074) [[ Website ]](https://nmsl.kaist.ac.kr/projects/sotta/)

## Installation Guide

1. Download or clone our repository.
2. Set up a Python environment using conda (see below).
3. Prepare datasets (see below).
4. Run the code (see below).

## Python Environment

We use [Conda environment](https://docs.conda.io/).
You can get conda by installing [Anaconda](https://www.anaconda.com/) first.

We share our Python environment that contains all required Python packages. Please refer to the `./sotta.yml` file.

You can import our environment using conda:

    conda env create -f sotta.yml -n sotta

Reference: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

## Prepare Datasets

To run our codes, you first need to download at least one of the datasets. Run the following commands:

    $ cd .                           #project root
    $ . download_cifar10c.sh        #download CIFAR10/CIFAR10-C datasets
    $ . download_cifar100c.sh       #download CIFAR100/CIFAR100-C datasets

Also, you can download the following datasets and locate them in the `./dataset` folder (create the folder if not exists):

- ImageNet-C: https://zenodo.org/record/2235448
- MNIST-C: https://zenodo.org/record/3239543

## Run

### Prepare Source model

"Source model" refers to a model that is trained with the source (clean) data only. Source models are required for all methods to perform test-time adaptation. 
We provide the pretrained model for CIFAR10/CIFAR100 with three random seeds (0,1,2) at [GDrive Link](https://drive.google.com/file/d/1PWKx5k5ePfw6XDgauPy-4J8riGXImzda/view?usp=sharing). After extracting `log.zip`, put this folder to the project root directory, i.e., `SoTTA/log`. 

Alternatively, you can train source models via:

    $ . train_src.sh                 #generate source models for CIFAR10 as default.

You can specify which dataset to use in the script file.

### Run Test-Time Adaptation (TTA)

Given source models are available, you can run TTA via:

    $ . tta.sh                       #run SoTTA for tta-target: CIFAR10-C, noisy-stream: MNIST as default.

You can specify which dataset and which method in the script file.

## Log

### Raw logs

In addition to console outputs, the result will be saved as a log file with the following structure: `./log/{DATASET}/{METHOD}_noisy/{TGT}/{LOG_PREFIX}_{SEED}_{DIST}/online_eval.json`

### Obtaining results

In order to print the classification accuracies(%) on the test set, run the following commands:

    $ python print_acc.py --method SoTTA    #prints the result of the specified condition.

## Tested Environment

We tested our codes in this environment.

- OS: Ubuntu 20.04.4 LTS
- GPU: NVIDIA GeForce RTX 3090
- GPU Driver Version: 470.74
- CUDA Version: 11.4

## Citation

```
@inproceedings{ gong2023sotta,
    title={{SoTTA}: Robust Test-Time Adaptation on Noisy Data Streams},
    author={Gong, Taesik and Kim, Yewon and Lee, Taeckyung and Chottananurak, Sorn and Lee, Sung-Ju},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
```
