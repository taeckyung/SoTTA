# SoTTA: Robust Test-Time Adaptation on Noisy Data Streams (NeurIPS '23)

This is the official PyTorch Implementation of 
"SoTTA: Robust Test-Time Adaptation on Noisy Data Streams (NeurIPS '23)" by 
[Taesik Gong*](https://taesikgong.com/), 
[Yewon Kim*](https://yewon-kim.com/), 
[Taeckyung Lee*](https://taeckyung.github.io/), 
Sorn Chottananurak, and 
[Sung-Ju Lee](https://sites.google.com/site/wewantsj/) (* Equal contribution).

[[ OpenReview ]](https://openreview.net/forum?id=3bdXag2rUd) [[ arXiv ]]() [[ Website ]](https://nmsl.kaist.ac.kr/projects/sotta/)

## Installation Guide

1. Download or clone our repository.
2. Set up a python environment using conda (see below).
3. Prepare datasets (see below).
4. Run the code (see below).

## Python Environment

We use [Conda environment](https://docs.conda.io/).
You can get conda by installing [Anaconda](https://www.anaconda.com/) first.

We share our python environment that contains all required python packages. Please refer to the `./sotta.yml` file

You can import our environment using conda:

    conda env create -f sotta.yml -n sotta

Reference: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

## Prepare Datasets

To run our codes, you first need to download at least one of the datasets. Run the following commands:

    $ cd .                           #project root
    $ . download_cifar10c.sh        #download CIFAR10/CIFAR10-C datasets
    $ . download_cifar100c.sh       #download CIFAR100/CIFAR100-C datasets

Also, you can download ImageNet-C at: https://zenodo.org/record/2235448 

## Run

### Prepare Source model

"Source model" refers to a model that is trained with the source (clean) data only. Source models are required to all methods to perform test-time adaptation. You can generate source models via:

    $ . train_src.sh                 #generate source models for CIFAR10 as default.

You can specify which dataset to use in the script file.

### Run Test-Time Adaptation (TTA)

Given source models are available, you can run TTA via:

    $ . tta.sh                       #Run SoTTA for tta-target: CIFAR10-C, noisy-stream: MNIST as default.

You can specify which dataset and which method in the script file.

## Log

### Raw logs

In addition to console outputs, the result will be saved as a log file with the following structure: `./log/{DATASET}/{METHOD}_noisy/{TGT}/{LOG_PREFIX}_{SEED}_{DIST}/online_eval.json`

### Obtaining results

In order to print the classification accuracies(%) on test set, run the following commands:

    $ python print_acc.py --dataset cifar10noisy --method SoTTA --seed 0 1 2    #print the result of the specified condition.

## Tested Environment

We tested our codes under this environment.

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
