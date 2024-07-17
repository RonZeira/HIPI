"""
File: cycif_eval_main.py
Author: Ron Zeira
Description: Script for running inference on a trained HIPI model: tile-->marker expression.
Usage example: python cycif_eval_main.py --cfg_file configs/ssl_vit_mlp8_16channels.yaml --ckpt_file "${ckpt_file}" --test_csv "${test_csv}" --num_workers 4 --out_path "${out_path}" --batch_size 512 --device cuda:0 --datasets "${dataset}"
"""

import os, sys, time, json, scipy, argparse, copy
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from code.data import *
from code.models import *
from code.utils import *

def exists(x):
    return not x is None

def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser()
    # Inputs/params
    parser.add_argument("--cfg_file", type=str, help="Configuration file", required=True)
    parser.add_argument("--ckpt_file", type=str, help="Model checkpoint", required=True)
    parser.add_argument("--test_csv", type=str, help="Test csv file", default = None)
    parser.add_argument("--batch_size", type=int, help="Override batch size", default = None)
    parser.add_argument("--num_workers", type=int, help="Override num worker", default = None)
    parser.add_argument("--datasets", type=str, nargs="+", help="Which datasets to predict (deault all)", default = None)
    parser.add_argument("--device", type=str, help="Device to use", default = 'cpu')
    parser.add_argument("--sample_prefix", type=str, help="Sample prefix str", default = None)
    # Outputs
    parser.add_argument("--out_path", type=str, help="Path to the output folder", default='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    print(args)
    
    config = OmegaConf.load(args.cfg_file)
    
    if exists(args.sample_prefix):
        config['data']['params']['common_args']['sample_prefix'] = args.sample_prefix
    config['data']['params']['train']['target'] = config['data']['params']['validation']['target']
    config['data']['params']['shuffle_test_dataloader'] = False
    if exists(args.test_csv):
        config['data']['params']['test'] = copy.deepcopy(config['data']['params']['validation'])
        config['data']['params']['test']['params']['df_file'] = args.test_csv
    if exists(args.batch_size):
        config['data']['params']['batch_size'] = args.batch_size
    if exists(args.num_workers):
        config['data']['params']['num_workers'] = args.num_workers
    print(config)
    
    model, _ = load_model(config, args.ckpt_file)
    datasets = instantiate_from_config(config.data)
    datasets.setup()
    
    if exists(args.datasets):
        dataset_names = args.datasets
    else:
        dataset_names = list(datasets.datasets.keys())
    
    print([(k, len(datasets.datasets[k]), len(datasets.datasets[k])/config['data']['params']['batch_size']) for k in dataset_names])
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    model = model.to(args.device)
    all_predictions = {}
    prediction_mats = {}
    with torch.no_grad():
        for n in dataset_names:
            i = 0
            prev_pct = 0
            all_predictions[n] = []
            start = time.time()
            for imgs, labels in datasets.dataloaders[n]():
                imgs = imgs.to(args.device)
                out = model(imgs)
                all_predictions[n].append(out.cpu().numpy())
                pct = int((config['data']['params']['batch_size']*len(all_predictions[n])/len(datasets.datasets[n])) * 100)
                if pct - prev_pct > 0:
                    print(n, config['data']['params']['batch_size']*len(all_predictions[n])/len(datasets.datasets[n]), time.time() - start)
                prev_pct = pct
                # if i > 50:
                # if pct >= 3:
                #     break
                # i+=1
        prediction_mats[n] = np.concatenate(all_predictions[n], axis=0)
        print(n, prediction_mats[n].shape)
        with open(f'{args.out_path}/{n}_predictions.npy', 'wb') as f:
            np.save(f, prediction_mats[n])
    
if __name__ == "__main__":
    main()