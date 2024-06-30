import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool

predicted_proteins = ['Keratin', 'Ki67', 'CD3', 'CD20', 'CD45RO', 'CD4', 'CD8a', 'CD68', 'CD163', 'FOXP3', 'PD1', 'PDL1', 'CD31', 'aSMA', 'Desmin', 'CD45']

def sample_corrs_from_dfs(dfs, cols, save = None, corr_func = scipy.stats.pearsonr,
                          top_q = None, middle_q = None, bottom_q = None):
    measured_df, predicted_df = dfs
    res_df = pd.DataFrame(columns=['Sample'] + cols)
    for sample in pd.unique(measured_df.Sample):
        measured_sample_df, predicted_sample_df = measured_df[measured_df.Sample==sample], predicted_df[predicted_df.Sample==sample]
        corrs = []
        for i in range(len(cols)):
            c = cols[i]
            measured_vec, predicted_vec = measured_sample_df[c].to_numpy(), predicted_sample_df[c].to_numpy()
            idx = None
            if not top_q is None:
                idx = np.where(measured_vec>=np.quantile(measured_vec, 1-top_q))[0]
            elif not middle_q is None:
                idx = np.where((measured_vec<=np.quantile(measured_vec, 1-middle_q)) & (measured_vec>=np.quantile(measured_vec, middle_q)))[0]
            elif not bottom_q is None:
                idx = np.where(measured_vec<=np.quantile(measured_vec, bottom_q))[0]
            if not idx is None:
                measured_vec, predicted_vec = measured_vec[idx], predicted_vec[idx]
            corr = corr_func(measured_vec, predicted_vec)[0]
            corrs.append(corr)
        res_df.loc[len(res_df.index)] = [sample] + corrs
    res_df['Sample'] = res_df['Sample'].astype('str')
    if not save is None:
        res_df.to_csv(save, index = False)
    return res_df

def plot_scatter_from_df(df, c, sample = None, axis=True, xy = ('x','y'), c_suffix = '.mean', figsize = (5,5), 
                         s=0.1, marker = 'o', show = True, norm = None, zscore = False, df_for_delta = None, 
                         cmap = 'viridis', colorbar = False, save = None, orientation='vertical', rotate_cbar_ticks = None):
    if sample is None:
        sample_df = df
    else:
        sample_df = df[df.Sample==sample]
    W = sample_df[list(xy)].to_numpy()
    z = sample_df[f'{c}{c_suffix}'].to_numpy()
    if zscore:
        z = scipy.stats.zscore(z)
        # z[(z<1) & (z>-1)] = 0
    if not df_for_delta is None:
        sample_df_for_delta = df_for_delta[df_for_delta.Sample==sample]
        z_for_delta = sample_df_for_delta[f'{c}{c_suffix}'].to_numpy()
        z_for_delta = scipy.stats.zscore(z_for_delta) if zscore else z_for_delta
        z = z - z_for_delta
        # z = scipy.stats.zscore(z)
        print(z.min(), z.max())
    plt.figure(figsize = figsize)
    if type(norm) == int or type(norm) == float:
        vmin = np.percentile(z, norm)
        vmax = np.percentile(z, 100 - norm)
        print(vmin , vmax)
    elif type(norm) == list or type(norm) == tuple:
        vmin , vmax = norm
    else:
        vmin = vmax = None
    if not df_for_delta is None:
        vmin, vmax = -3, 3
    # cmap = 'bwr'
    g = plt.scatter(W[:,1], W[:,0], c=z, cmap=cmap, s=s, marker = marker, vmin = vmin, vmax = vmax, linewidth=0)
    plt.gca().invert_yaxis()
    if not axis:
        plt.gca().axis('off')
    if colorbar:
        cbar = plt.colorbar(orientation=orientation, pad=-0.03, shrink=0.8, aspect = 30, format = '%1.1f')
        cbar.ax.tick_params(labelsize=15)
        if not rotate_cbar_ticks is None:
            cbar.ax.tick_params(rotation=rotate_cbar_ticks)
    if not save is None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
        
def plot_diff_hist(df1, df2, sample, c, xy = ('x','y'), c_suffix = '.mean', figsize = (5,5), zscore = True, first_name = '1', second_anme = '2'):
    sample_df1 = df1[df1.Sample==sample]
    W = sample_df1[list(xy)].to_numpy()
    z1 = sample_df1[f'{c}{c_suffix}'].to_numpy()
    sample_df2 = df2[df2.Sample==sample]
    z2 = sample_df2[f'{c}{c_suffix}'].to_numpy()
    if zscore:
        z1 = scipy.stats.zscore(z1)
        z2 = scipy.stats.zscore(z2)
    delta = z1-z2
    print(delta.mean(), delta.std())
    sns.histplot(delta, kde=True)
    sns.histplot(z1, kde=True)
    sns.histplot(z2, kde=True)
    plt.legend(['difference', first_name, second_anme])
    plt.show()
    

def get_knn(df, k = 1, xy_name = ['x', 'y'], algorithm='auto'):
    XY = df[xy_name].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(XY)
    distances, indices = nbrs.kneighbors(XY)
    return distances, indices 

def spatial_corrs_from_dfs(df1, col1, k = 1, df2 = None, col2 = None, xy_name = ['x', 'y'], c_suffix = '.mean', distances = None, indices = None):
    df2 = df1 if df2 is None else df2
    col2 = col1 if col2 is None else col2
    if indices is None:
        distances, indices = get_knn(df1, k = k, xy_name = xy_name)
    A, B = df1[col1+c_suffix].to_numpy(), df2[col2+c_suffix].to_numpy() 
    A, B = scipy.stats.zscore(A), scipy.stats.zscore(B)
    corrs = A.dot(B[indices])/len(A)
    return corrs

def plot_multiple_corrs(corrs_list, line_args_list = None, title = ''):
    k = len(corrs_list[0])
    for i in range(len(corrs_list)):
        plt.plot(np.arange(k), corrs_list[i], **line_args_list[i])
    plt.ylim([0,1])
    # col_str = f'{col1}' if col1==col2 else f'{col1}~{col2}'
    plt.title(title)
    plt.xlabel('nearest neighbor')
    plt.ylabel('correlation')
    plt.legend()
    plt.show()
    
def plot_binarized_contour(df, prot, suffix = '.mean', cont = True, class_idx = 1, total_classes = 4, figsize = (6,6), palette="deep", linewidth=0, s=2.0):
    plt.figure(figsize=figsize)
    hue_order = np.arange(0,1,1.0/class_idx).tolist() + np.arange(1,total_classes-class_idx+1,1).tolist()
    sns.scatterplot(data=df, x="y", y="x", hue=prot+suffix, palette=palette,linewidth=linewidth,s=s, hue_order = hue_order, legend = False)
    plt.gca().invert_yaxis()
    plt.gca().axis('off')
    if cont:
        sns.kdeplot(data=df[df[prot+suffix] == 1], x="y", y="x", 
            hue=prot+suffix, palette=palette, hue_order = hue_order, legend = False)
    plt.show()
    

from sklearn.mixture import GaussianMixture
def fit_gmm(v, **kwargs):
    gm = GaussianMixture(n_components=2, random_state=0, max_iter=1000, n_init=10, **kwargs).fit(v.reshape(-1, 1))
    a = gm.means_.min()
    b = gm.means_.max()
    l = np.linspace(min(v), max(v)+1.0/len(v), num = len(v)+1)
    c = gm.predict(l.reshape(-1,1))
    d = np.abs(c[:-1]-c[1:])
    # print(sum(d))
    if sum(d) == 1:
        e = np.where(d!=0)[0][0]
        f = (l[e]+l[e+1])/2
    elif sum(d) == 2:
        # e = int(np.mean(np.where(d!=0)[0]))
        # f = (l[e]+l[e+1])/2
        min_i = np.where(gm.means_ == a)[0][0]
        max_i = np.where(gm.means_ == b)[0][0]
        var_min = gm.covariances_[min_i,0,0]
        var_max = gm.covariances_[max_i,0,0]
        f = (a*var_max + b*var_min)/(var_min+var_max)
        # f = (a+b)/2
    else:
        f = (a+b)/2
    return a, b, f, gm

def plot_gmm_expression(df, cols = predicted_proteins, col_sffix = '.mean'):
    fig, axs = plt.subplots(1, len(cols), figsize=(15, 4))
    qs = []
    for i, g in enumerate(cols):
        v = df[g+col_sffix].to_numpy()
        sns.histplot(v, kde=True, ax=axs[i]).set(title=g)
        a, b, f, gm = fit_gmm(v)
        axs[i].axvline(x = a, color = 'r')
        axs[i].axvline(x = b, color = 'g')
        axs[i].axvline(x = f, color = 'k')
        y_min, y_max = axs[i].get_ylim()
        l = np.linspace(min(v), max(v)+1.0/len(v), num = len(v)+1)
        c = gm.predict(l.reshape(-1,1))
        axs[i].plot(l, c*y_max, color = 'y')
        qs.append(sum(v>f)/len(v))
    # print(qs)
    # fig.suptitle(sample)
    plt.show()  

def quantize_expression_df(df, cols = predicted_proteins, col_sffix = '.mean' , ths = None):
    df_copy = df.copy()
    ths_ = ths or []
    for i, g in enumerate(cols):
        v = df[g+col_sffix].to_numpy()
        if ths is None:
            _, _, th, _ = fit_gmm(v)
            ths_.append(th)
        else:
            th = ths[i]
        y = (v>th)*1.0
        df_copy[g+col_sffix] = y
    return df_copy, ths_

def quantize_all_samples(df, parallel = False, limit = None):
    samples = pd.unique(df.Sample) if limit is None else pd.unique(df.Sample)[:limit]
    if not parallel:
        df_quant = pd.concat([
            quantize_expression_df(df[df.Sample == sample])[0] for sample in samples
        ])
    else:
        processes = len(samples)
        with Pool(processes = processes) as pool:
            res = pool.map(quantize_expression_df, [df[df.Sample == sample] for sample in samples])
        df_quant = pd.concat([res[i][0] for i in range(len(samples))])
    df_quant = df_quant.rename(columns={f'{p}.mean':f'{p}' for p in predicted_proteins})
    df_quant[predicted_proteins] = df_quant[predicted_proteins].astype(bool)
    return df_quant

def plot_prot_cooccurrence(df, prots, figsize=(8,8), s = 2.0, palette = 'deep', others_str = 'Others', save = None , hue_order = None, legend = True):
    color_map = isinstance(palette, dict)
    df_copy = df.copy()
    joint_str = '/'.join(prots)
    if len(prots) == 1:
        joint_str += ' '
    df_copy[joint_str] = (df_copy[prots] * np.array([2**i for i in range(len(prots))])).sum(axis=1)
    n = len(prots)
    d = {}
    for j in range(2**len(prots)):
        curr_str = []
        for i in range(n):
            if f'{j:{n}b}'[n-i-1] == '1': curr_str.append(f'{prots[i]}+')
        curr_str = others_str if len(curr_str)==0 else '/'.join(curr_str)
        curr_str = others_str if color_map and (not curr_str in palette) else curr_str
        d[j] = curr_str
    df_copy[joint_str] = df_copy[joint_str].replace(d)
    if not color_map:
        hue_order = [d[j] for j in range(2**len(prots))]
    plt.figure(figsize=figsize)
    g = sns.scatterplot(data=df_copy, x="y", y="x", hue=joint_str, palette=palette, linewidth=0, s = s, hue_order = hue_order, legend = legend)
    plt.gca().invert_yaxis()
    plt.gca().axis('off')
    if legend:
        sns.move_legend(g, "upper left", title=None, frameon=True, fontsize = 13.5, ncol = 2, columnspacing = 0.5, labelspacing = 0.0, handlelength = 0.0, bbox_to_anchor=(0.0, 0.0))# bbox_to_anchor=(0.0, 1.0))
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    return df_copy