import json
import os
import re
import argparse

import pandas as pd
from multiprocessing import Pool
from functools import partial

CORRUPTION_LIST = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"]

METHOD_LIST = ["Src", "TENT", "PseudoLabel", "BN_Stats", "SAR", "RoTTA", "CoTTA", "LAME", "MEMO", "SoTTA", "EATA"]

BASE_DATASET = "cifar10noisy"

LOG_PREFIX = "eval_results_neurips_final"

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
                for noise in args.noise:
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


def main(args):
    root = 'log/' + args.dataset
    paths = [os.path.join(root, f"{method}_noisy") for method in args.method]
    with Pool(processes=len(paths)) as p:
        func = partial(process_path, args)
        results = p.map(func, paths)

    for noisy_type in args.noisy_type:
        for seed in args.seed:
            print(f"SEED:{seed}, NOISY_TYPE: {noisy_type}")
            result = pd.concat([results[i][f"{seed}_{noisy_type}"] for i in range(len(results))])
            print(result.to_csv())


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
        f"Noise: {args.noise}\n"
        f"Dist: {args.dist}"
    )

    main(args)
