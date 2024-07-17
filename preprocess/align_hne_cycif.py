"""
File: utils.py
Author: Ron Zeira
Description: script for aligning HnE and Cycif images
Usage example: python align_hne_cycif.py --hne /home/jupyter/Data/CRC_Lin/WD-76845-006.ome.tif --cycif /home/jupyter/Data/CRC_Lin/WD-76845-007.ome.tif --cells /home/jupyter/Data/CRC_Lin/WD-76845-007.csv --global_affine /home/jupyter/CycifPreprocess/ProcessedData/WD-76845-007.ome_affine_transform.npy --new_coordinates /home/jupyter/CycifPreprocess/ProcessedData/WD-76845-007.ome_new_coordinates.npy --patch_size 16384 --slack 512 --stride 1024 --resize_registration 0.03125 --min_spots 5000
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
# sys.path.append("STalign/STalign/")
# import STalign
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
    ## Reading
    parser.add_argument("--resize_ratio_hne", type=float, default=1.0, help="Resize ratio for HnE image")
    parser.add_argument("--cycif_to_hne_ratio", type=float, default=2.0, help="Cycif to HnE resize ratio")
    parser.add_argument("--hne_grayscale", type=str2bool, default=True, help="Grayscale HnE")
    parser.add_argument("--hne_invert_color", type=str2bool, default=True, help="Invert color HnE HnE")
    parser.add_argument("--cycif_image_channel", type=int, default=0, help="Cycif image channel number for alignment")
    parser.add_argument("--cycif_rotate_90clockwise", type=str2bool, default=False, help="Rotate cycif 90 degree clockwise")
    parser.add_argument("--cycif_log_transform", type=str2bool, default=True, help="Cycif log transform channel values")
    parser.add_argument("--cycif_min_max_norm", type=str2bool, default=True, help="Cycif channel min max norm")
    parser.add_argument("--cycif_quantile_norm", type=float, default=0.01, help="Cycif channel quantile norm")
    parser.add_argument("--cycif_cells_channel", type=str, default='Hoechst0', help="Cycif cell channel name")
    ## Global alignment
    parser.add_argument("--global_alignment_resize", type=float, default=0.01, help="Global alignment resize")
    ## Local alignment
    parser.add_argument("--patch_size", type=int, default=2**14, help="Local alignment patch size")
    parser.add_argument("--slack", type=int, default=512, help="Local alignment slack for coordinates inside a patch")
    parser.add_argument("--stride", type=int, default=2**10, help="Local alignment stride for patches")
    parser.add_argument("--resize_registration", type=float, default=1.0/32, help="Local alignment resize registration")
    parser.add_argument("--min_spots", type=float, default=2000, help="Local alignment minimum spots for translation")
    
    # Inputs
    parser.add_argument("--hne", type=str, help="Path to the HnE input file", required=True)
    parser.add_argument("--cycif", type=str, help="Path to the Cycif image input file", required=True)
    parser.add_argument("--cells", type=str, help="Path to the cells input file", required=True)
    
    #outputs
    parser.add_argument('-g', "--global_affine", type=str, help="Path to the global translation output file", required=True)
    parser.add_argument('-n', "--new_coordinates", type=str, help="Path to the new coordinates output file", required=True)
    parser.add_argument("--save_new_coordinates_csv", type=str2bool, default=True, help="Save new coordinates csv")
    parser.add_argument("--save_point_coverage_counts", type=str2bool, default=True, help="save_point_coverage_counts")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Read the input file.
    print('read_hne_from_tif')
    hne_image = read_hne_from_tif(args.hne, grayscale = args.hne_grayscale, resize = args.resize_ratio_hne, invert_color = args.hne_invert_color)
    print('HNE shape', hne_image.shape)
    
    print('read_cycif_channel_from_tif')
    cycif_image = read_cycif_channel_from_tif(args.cycif, resize = args.cycif_to_hne_ratio * args.resize_ratio_hne, channel = args.cycif_image_channel, 
                                          log_transform = args.cycif_log_transform, min_max_norm = args.cycif_log_transform, 
                                              quantile_norm = args.cycif_quantile_norm, rotate_90clockwise = args.cycif_rotate_90clockwise)
    print('Cycif shape', cycif_image.shape)
    
    print('read_coordinates_file')
    W, z = rotate_coordinates_file(args.cells, image_shape = args.cycif, 
                               final_resize = args.cycif_to_hne_ratio * args.resize_ratio_hne, 
                                   channel = args.cycif_cells_channel, rotate_90clockwise = args.cycif_rotate_90clockwise)
    
    # Global alignment
    print('global_affine_registration_elastix')
    T = global_affine_registration_elastix(hne_image, cycif_image, resize_ratio = args.global_alignment_resize)
    print('Affine tranform', T)
    np.save(args.global_affine, T)
    
    # Global affine transform
    print('affine_transformation_image')
    cycif_image = affine_transformation_image(cycif_image, T.copy(), output_shape = hne_image.shape[:2], translation_resize = None, 
                                              image_resize = args.cycif_to_hne_ratio * args.resize_ratio_hne, 
                                              rotation_resize_ratio = args.cycif_to_hne_ratio * args.global_alignment_resize)
    print('Cycif shape after rotation', cycif_image.shape)
    
    print('affine_transformation_coordinates')
    W = affine_transformation_coordinates(W, T, image_shape = hne_image.shape[:2], translation_resize = None, 
                                          image_resize = args.cycif_to_hne_ratio * args.resize_ratio_hne, 
                                              rotation_resize_ratio = args.cycif_to_hne_ratio * args.global_alignment_resize)
    
    # local alignment
    print('correct_coordinates_with_patch_alignment')
    corrected_coordinates , count_points = correct_coordinates_with_patch_alignment(hne_image, cycif_image, W, patch_size = args.patch_size, 
                                                                                    stride = args.stride, slack = args.slack, 
                                                                                plot = False, resize_registration = args.resize_registration, 
                                                                                    min_spots = args.min_spots)
    
    # Write results
    print('Write results')
    np.save(args.new_coordinates, corrected_coordinates)
    if args.save_point_coverage_counts:
        i = args.new_coordinates.rfind(".")
        count_file = args.new_coordinates[:i] + '_point_count' + args.new_coordinates[i:]
        np.save(count_file, count_points)
    if args.save_new_coordinates_csv:
        i = args.cells.rfind(".")
        new_cell_file = args.cells[:i] + '_new_coordinates' + args.cells[i:]
        df = pd.read_csv(args.cells)
        df[['hne_X','hne_Y']] = corrected_coordinates
        df.to_csv(new_cell_file, index = False)
        
if __name__ == "__main__":
    main()