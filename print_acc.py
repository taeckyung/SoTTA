import json
import os
import re
import argparse

import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from texttable import Texttable

CORRUPTION_LIST = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"]

CORRUPTION_DICT = {
    "gaussian_noise": "Gau.", 
    "shot_noise": "Shot", 
    "impulse_noise": "Imp.", 
    "defocus_blur": "Def.", 
    "glass_blur": "Gla.", 
    "motion_blur": "Mot.",
    "zoom_blur": "Zoom", 
    "snow": "Snow", 
    "frost": "Fro.", 
    "fog": "Fog", 
    "brightness": "Brit.", 
    "contrast": "Cont.", 
    "elastic_transform": "Elas.", 
    "pixelate": "Pix",
    "jpeg_compression": "JPEG"
}

METHOD_LIST = ["Src", "TENT", "PseudoLabel", "BN_Stats", "SAR", "RoTTA", "CoTTA", "LAME", "MEMO", "SoTTA", "EATA"]

BASE_DATASET = "cifar10noisy"

LOG_PREFIX = "eval_results"

SEED = [0, 1, 2]

DIST = 1

NOISE = ["original", "cifar100", "imagenet", "mnist", "uniform", "repeat"]


def get_avg_online_acc(file_path):
    f = open(file_path)
    json_data = json.load(f)
    f.close()
    return json_data['accuracy'][-1]


def process_path(args, path):
    result = {f"{s}_{t}": pd.DataFrame(columns=CORRUPTION_LIST) for s in args.seed for t in args.noisy_type}
    method = path.split("/")[-1].replace("_noisy", "")
    for (path, _, _) in os.walk(path):
        for corr in CORRUPTION_LIST:
            for seed in args.seed:
                for noise in args.noisy_type:
                    if method == 'Src':
                        pattern_of_path = f'.*{corr}.*{noise}.*/.*{args.prefix}_.*{seed}.*'
                    elif 'repeat' in noise:  # attack
                        if args.dataset in ['cifar10noisy', 'cifar100noisy']:
                            attack = "tta_attack_indiscriminate_num20_step10_eps0.1"
                        elif args.dataset == 'imagenetnoisy':
                            attack = "tta_attack_indiscriminate_num20_step1_eps0.2"
                        else:
                            raise NotImplementedError
                        pattern_of_path = f'.*{corr}.*{noise}.*/{attack}/.*{args.prefix}_.*{seed}_dist{args.dist}.*'
                    else:
                        pattern_of_path = f'.*{corr}.*{noise}.*/.*{args.prefix}_.*{seed}_dist{args.dist}.*'

                    pattern_of_path = re.compile(pattern_of_path)
                    if pattern_of_path.match(path):
                        if not path.endswith('/cp'):  # ignore cp/ dir
                            try:
                                acc = get_avg_online_acc(path + '/online_eval.json')
                                key = method + "_" + path.split("/")[-1]
                                result[f"{seed}_{noise}"].loc[key, corr] = float(acc)
                            except:
                                pass
    return result

def pretty_print(results, noise_list, seed_list):
    print(f'Classification accuracy(%) of method {args.method[0]}\n')

    t = Texttable(max_width=160)
    t.set_precision(1)
    t.set_deco(t.HEADER)


    for i, noise_type in enumerate(noise_list):
        if len(seed_list) > 1:
            accs = []

        for j, seed in enumerate(seed_list):

            if i == 0 and j == 0:
                t_head = ["Noise", "Seed"] + [CORRUPTION_DICT[k] for k in list(results[0][f"{seed}_{noise_type}"].columns)] + ["AVG"]
                t.header(t_head)
            
            if j == 0:
                acc_values = [noise_type, seed]
            else:
                acc_values = ["", seed]

            result = pd.concat([results[i][f"{seed}_{noise_type}"] for i in range(len(results))]).values
            if len(result) == 0:
                continue

            acc_values += list(result[0])
            acc_values += [np.mean(result[0])]
            t.add_row(acc_values)

            accs.append(result[0])
        
        # avg
        if len(seed_list) > 1 and len(accs) > 1:
            accs = np.asarray(accs)
            acc_mean = np.mean(accs, axis=0).tolist()
            acc_values = ["", "AVG"] + acc_mean + [np.mean(acc_mean)]
            t.add_row(acc_values)

    print(t.draw())
    print('\n')

    

def main(args):

    print("Processing data logs...\n")

    root = 'log/' + args.dataset
    paths = [os.path.join(root, f"{method}_noisy") for method in args.method]

    with Pool(processes=len(paths)) as p:
        func = partial(process_path, args)
        results = p.map(func, paths)

    pretty_print(results, args.noisy_type, args.seed)

def parse_arguments():
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=BASE_DATASET,
                        help='Base dataset')

    parser.add_argument('--method', nargs="*", type=str, default=METHOD_LIST,
                        help='Method name')

    parser.add_argument('--seed', nargs="*", type=int, default=SEED,
                        help='Seed')

    parser.add_argument('--noisy_type', nargs="*", type=str, default=NOISE,
                        help='Noisy data type')

    parser.add_argument('--prefix', type=str, default=LOG_PREFIX,
                        help='Log prefix')

    parser.add_argument('--dist', type=str, default=DIST,
                        help='Distribution')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print(
        f"DATASET: {args.dataset}\n"
        f"LOG_PREFIX: {args.prefix}\n"
        f"METHOD: {args.method}\n"
        f"SEED: {args.seed}\n"
        f"Noise: {args.noisy_type}\n"
        f"Dist: {args.dist}"
    )

    main(args)
