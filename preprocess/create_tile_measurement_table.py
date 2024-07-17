"""
File: create_tile_measurement_table.py
Author: Ron Zeira
Description: script for creating a tile feature dataframe table
### Set constantsand arguments below
"""

import numpy as np
import pandas as pd
from preprocess_funcs import *
import itertools

###########################################################################################################################
####### FUNCTIONS FOR ALIGNING CYCIF TO HNE AND GETTING PATCH EXPRESSIONS ################################################
###########################################################################################################################

def idx_dict_to_array(d, shape):
    return scipy.sparse.coo_array((list(d.values()),(list(map(lambda t: t[0],d.keys())), list(map(lambda t: t[1],d.keys())))), shape = shape).todense()

def idx_dict_to_array_multi(d, shape):
    return np.concatenate([idx_dict_to_array(dict(map(lambda t: (t[0], t[1][i]), d.items())), shape = shape[:-1]).reshape((*shape[:-1],1)) for i in range(shape[-1])],axis=2)

def get_patch_cell_dict(im_shape, coordinates, patch_size = 256, stride = 128):
    out_shape = np.array(np.floor((im_shape - patch_size)/stride)+1, dtype=int)
    out_shape_rev = out_shape[::-1]
    cells_dict = {}
    max_ji = np.minimum(np.floor((coordinates)/stride).astype(int) + 1, out_shape_rev)
    min_ji = np.maximum(np.floor((coordinates - patch_size)/stride).astype(int) + 1, 0)
    for k in range(len(coordinates)):
        for j in range(min_ji[k,0], max_ji[k,0]):
            for i in range(min_ji[k,1], max_ji[k,1]):
                d = cells_dict.get((i,j), [])
                d.append(k)
                cells_dict[i, j] = d        
    return cells_dict

def get_patch_expression(im_shape, coordinates, values, patch_size = 256, stride = 128, min_points = 1, vmin = None, return_mats = False):
    cells_dict = get_patch_cell_dict(im_shape, coordinates, patch_size = patch_size, stride = stride)
    out_shape = np.array(np.floor((im_shape - patch_size)/stride)+1, dtype=int)
    vmin = vmin if vmin else values.min(axis=0)
    
    if min_points>1:
        cells_dict = dict(filter(lambda pair: len(pair[1])>=min_points, cells_dict.items()))

    cell_means = dict(map(lambda t: (t[0], values[t[1]].mean(axis=0)), cells_dict.items()))
    cell_vars = dict(map(lambda t: (t[0], values[t[1]].var(axis=0)), cells_dict.items()))
    cell_counts = dict(map(lambda t: (t[0], len(t[1])), cells_dict.items()))
    
    cell_counts_mat = idx_dict_to_array(cell_counts, out_shape)
    cell_means_mat = idx_dict_to_array_multi(cell_means, (*out_shape,values.shape[-1]))
    cell_vars_mat = idx_dict_to_array_multi(cell_vars, (*out_shape,values.shape[-1]))
    cell_means_mat[cell_counts_mat<min_points] = vmin
    cell_vars_mat[cell_counts_mat<min_points] = 0
    
    if return_mats:
        return cells_dict, (cell_counts, cell_means, cell_vars), (cell_counts_mat, cell_means_mat, cell_vars_mat)
    else:
        return cells_dict, (cell_counts, cell_means, cell_vars)
    
def samples_patch_agg_to_df(dictionary, samples, channels = ['Keratin', 'Ki67', 'CD3', 'CD20', 'CD45RO', 'CD4', 'CD8a', 'CD68', 'CD163', 'FOXP3', 'PD1', 'PDL1', 'CD31', 'aSMA', 'Desmin', 'CD45'], 
                            only_with_cells = True, step = 128, patch_size = 256):
    all_samples_patch_df = pd.DataFrame(columns = ['Sample', 'x', 'y', 'cells']+[c+'.mean' for c in channels] + [c+'.var' for c in channels])
    mat_suffix = '' if only_with_cells else "_mat"
    for sample in samples:
        print(sample)
        cell_counts, cell_means, cell_vars = dictionary[sample]['cell_counts'+mat_suffix], dictionary[sample]['cell_means'+mat_suffix], dictionary[sample]['cell_vars'+mat_suffix]
        n = len(cell_counts) if only_with_cells else cell_counts.shape[0] * cell_counts.shape[1]
        # sample_vec = np.ones((n,1)) * sample
        sample_vec = np.array([sample] * n).reshape((n,1))
        if only_with_cells:
            coo_idx = np.array(list(cell_means.keys()))*step
            cell_counts_mat = np.array(list(cell_counts.values())).reshape(n,1)
            cell_means_mat = np.array(list(cell_means.values()))
            cell_vars_mat = np.array(list(cell_vars.values()))
        else:
            coo_idx = np.array(list(itertools.product(range(cell_counts.shape[0]), range(cell_counts.shape[1])))) * step
            cell_counts_mat = cell_counts.flatten().reshape((len(coo_idx), 1))
            cell_means_mat = cell_means.flatten().reshape((len(coo_idx), cell_means.shape[-1]))
            cell_vars_mat = cell_vars.flatten().reshape((len(coo_idx), cell_vars.shape[-1]))
        all_patches_of_sample  = np.concatenate((sample_vec, coo_idx, cell_counts_mat, cell_means_mat, cell_vars_mat), axis = 1)
        all_patches_of_sample = pd.DataFrame(all_patches_of_sample, columns=all_samples_patch_df.columns)
        # all_samples_patch_df = all_samples_patch_df.append(all_patches_of_sample, ignore_index=True)
        all_samples_patch_df = pd.concat([all_samples_patch_df, all_patches_of_sample], ignore_index= True)
    all_samples_patch_df = all_samples_patch_df.astype({'Sample':str, 'x':int, 'y':int, 'cells':int, **{c+'.mean':float for c in channels}, **{c+'.var':float for c in channels}})
    return all_samples_patch_df

def split_df_train_test(df, pct = (0.5, 0.5), which = 1):
    samples = sorted(pd.unique(df.Sample))
    middle_dict = {s:df[df.Sample==s][['x','y']].to_numpy().min(axis=0)+np.array(pct)*(df[df.Sample==s][['x','y']].to_numpy().max(axis=0)-df[df.Sample==s][['x','y']].to_numpy().min(axis=0)) for s in samples}
    which_quart_sign = [np.array([1,1]), np.array([1,-1]), np.array([-1,1]), np.array([-1,-1])][which]
    bla = df.apply(lambda row : (which_quart_sign[0]*(row['x']-middle_dict[row['Sample']][0])>=0) and 
                   (which_quart_sign[1]*(row['y']-middle_dict[row['Sample']][1])>=0), axis = 1)
    return df[~bla], df[bla]

###########################################################################################################################
######### SET CONSTANTS AND PARSE ARGUMENTS ##############################################################################
###########################################################################################################################
channels = ['Keratin', 'Ki67', 'CD3', 'CD20', 'CD45RO', 'CD4', 'CD8a', 'CD68', 'CD163', 'FOXP3', 'PD1', 'PDL1', 'CD31', 'aSMA', 'Desmin', 'CD45']
coordinate_names = ['hne_X', 'hne_Y']
min_points = 1
patch_size = 256
step = 128
data_path = '/home/jupyter/Data/CRC_Lin/'
output_path = '/home/jupyter/CycifPreprocess/ProcessedData/'
potential_samples = {f'{i:02}':f'{data_path}/CRC{i:02}_new_coordinates.csv' for i in [2, 3, 12, 13, 14, 15, 17]}

###########################################################################################################################
######### READ CELL COORDINATES AND EXPRESSION DATA FOR EACH SAMPLE #############################
###########################################################################################################################

dfs = []
for sample in potential_samples:
    df = pd.read_csv(potential_samples[sample])
    df_ = df[channels].copy()
    df_ = np.log(df_ + 1)
    df_ = (df_ - df_.mean())/df_.std()
    df_[channels] = df_[channels].to_numpy()
    df_[coordinate_names] = df[coordinate_names].to_numpy()
    df_['Sample'] = f'CRC{sample}'
    dfs.append(df_)
all_df = pd.concat(dfs)

###########################################################################################################################
######### CALCULATE PATCH EXPRESSIONS FOR EACH SAMPLE #############################
###########################################################################################################################

all_samples_patch_agg = {}
for sample in potential_samples:
    sample = f'CRC{sample}'
    sub_df = all_df[all_df.Sample == sample]
    tif_file = tifffile.TiffFile(f'{data_path}/{sample}-HE.ome.tif')
    sample_shape = np.array(tif_file.pages[0].shape)
    coordinates = sub_df[coordinate_names].to_numpy()
    values = sub_df[channels].to_numpy()
    print(sample, sample_shape, coordinates.shape, values.shape)
    cells_dict, (cell_counts, cell_means, cell_vars), (cell_counts_mat, cell_means_mat, cell_vars_mat) = get_patch_expression(sample_shape, coordinates, values, patch_size = patch_size, stride = step, 
                                                                                  min_points = min_points, vmin = None, return_mats = True)
    all_samples_patch_agg[sample] = {'cell_counts':cell_counts, 'cell_means':cell_means, 'cell_vars':cell_vars, 
                                     'cell_counts_mat':cell_counts_mat, 'cell_means_mat':cell_means_mat, 'cell_vars_mat':cell_vars_mat}

all_samples_patch_df = samples_patch_agg_to_df(all_samples_patch_agg, [f'CRC{sample}' for sample in potential_samples], only_with_cells = True)
print(all_samples_patch_df.shape, all_samples_patch_df.dtypes)
# print(all_samples_patch_df.head())

all_samples_all_patch_df = samples_patch_agg_to_df(all_samples_patch_agg, [f'CRC{sample}' for sample in potential_samples], only_with_cells = False)
print(all_samples_all_patch_df.shape, all_samples_all_patch_df.dtypes)

all_samples_patch_df.to_csv(f'{output_path}/crc_sample_patches_measurements_with_cells_gating_channels.csv', index = False)
all_samples_all_patch_df.to_csv(f'{output_path}/crc_sample_patches_all_measurements_gating_channels.csv', index = False)


###########################################################################################################################
######### SPLIT TRAI/VAL/TEST #############################
###########################################################################################################################

sub_df0, sub_df1 = split_df_train_test(all_samples_patch_df, pct = (0.5, 0.5), which = 1)
sub_df1_0, sub_df1_1 = split_df_train_test(sub_df1, pct = (1.0, 0.5), which = 2)
sub_df0.to_csv(f'{output_path}/sample_patches_measurements_with_cells_gating_channels_train.csv', index = False)
sub_df1_1.to_csv(f'{output_path}/sample_patches_measurements_with_cells_gating_channels_val.csv', index = False)
sub_df1_0.to_csv(f'{output_path}/sample_patches_measurements_with_cells_gating_channels_test.csv', index = False)
