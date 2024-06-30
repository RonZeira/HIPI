import os, sys, gc, time, json, scipy, cv2, copy, skimage, tifffile, PIL
import numpy as np
import pandas as pd
from preprocess_funcs import *
import argparse

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
    # Defaults
    
    # Inputs
    parser.add_argument("--patch_size", type=int, default=256, help="patch_size")
    parser.add_argument("--step", type=int, default=128, help="step")
    parser.add_argument("--hne", type=str, help="Path to the HnE input file", required=True)
    # Outputs
    parser.add_argument("--out_path", type=str, help="Path to the output folder", default='')
    parser.add_argument("--out_file_prefix", type=str, help="out_file_prefix", default='')
    parser.add_argument("--out_file_suffix", type=str, help="out_file_suffix", default='png')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    hne_image = read_hne_from_tif(args.hne, grayscale = False, resize = 1, invert_color = False)   
    patches = view_as_windows(hne_image, args.patch_size, step=args.step)
    print(args.hne, hne_image.shape , patches.shape)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    total = patches.shape[0] * patches.shape[1]
    cummulative = 0
    start = time.time()
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            cummulative += 1
            file_name = f"{args.out_path}/{args.out_file_prefix}_{int(i*args.step)}_{int(j*args.step)}.{args.out_file_suffix}" if len(args.out_file_prefix)>0 else f"{args.out_path}/{int(i*args.step)}_{int(j*args.step)}.{args.out_file_suffix}"
            PIL.Image.fromarray((patches[i,j,0]* 255).astype(np.uint8)).save(file_name)
            if cummulative%10000 == 0:
                print(args.hne, float(cummulative)/total, time.time()-start)
    print(args.hne, 1.0, time.time()-start)

if __name__ == "__main__":
    main()