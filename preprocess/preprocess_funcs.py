import os, sys, scipy, cv2, tifffile, skimage
import xml.etree.ElementTree
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import AffineTransform
from skimage.transform import warp
# sys.path.append("STalign/STalign/")
# import STalign
import numbers
from numpy.lib.stride_tricks import as_strided

def resize_image_ratio(img, ratio = 1.0, interpolation = cv2.INTER_AREA):
    if ratio == 1.0: return img
    return cv2.resize(img, 
                    (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                    interpolation = interpolation)

def rotate_coordinates_file(W, image_shape = None, final_resize = 1.0, channel = None, rotate_90clockwise = True):
    # read dataframe and extract coordinates
    df = None
    if isinstance(W, str):
        df = pd.read_csv(W)
        W = df
    if isinstance(W, pd.DataFrame):
        df = W
        W = df[['Xt', 'Yt']].to_numpy()
    # get image size for rotation
    if isinstance(image_shape, str):
        tif_file = tifffile.TiffFile(image_shape)
        y_l, x_l = tif_file.pages[0].shape
    elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
        y_l, x_l = image_shape
    else:
        y_l, x_l = W.max(axis=0)
    # move coordinates to image scale
    cycif_to_feature_points_scale = 0.65 # percent of original cycif size
    W = W/cycif_to_feature_points_scale # coordinates in image scale
    # rotate
    if rotate_90clockwise:
        center = np.array([[1, 0, y_l/2], [0, 1, x_l/2], [0, 0, 1]] )
        center_inverse = np.array([[1, 0, -x_l/2], [0, 1, -y_l/2], [0, 0, 1]] )
        T = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]] )
        total_matrix = center @ T @ center_inverse
        W = total_matrix.dot(np.hstack((W,np.ones((W.shape[0],1)))).T).T[:,:2]
    # resize coordiantes
    W = W * final_resize
    if channel is None:
        return W
    else:
        return W, df[channel].to_numpy()
    
def affine_transformation_coordinates(W, T, image_shape = None, translation_resize = None, image_resize = 1.0, rotation_resize_ratio = 1.0):
    if T.shape[1] == 2:
        T = np.hstack((T,np.array([[0,0,1]]).T)).shape
    # update translation to current resize
    if translation_resize is None:
        translation_resize = image_resize/rotation_resize_ratio
    T[:2,-1] = T[:2,-1] * translation_resize
    if isinstance(image_shape, tuple) or isinstance(image_shape, list):
        h, w = image_shape
    else:
        h, w = W.max(axis=0)
    # center and rotate
    center = np.array([[1, 0, w/2], [0, 1, h/2], [0, 0, 1]] )
    center_inverse = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]] )
    total_matrix = center @ T @ center_inverse
    total_matrix = np.linalg.inv(total_matrix)
    W = total_matrix.dot(np.hstack((W,np.ones((W.shape[0],1)))).T).T[:,:2]
    return W

def affine_transformation_image(img, T, output_shape = None, translation_resize = None, image_resize = 1.0, rotation_resize_ratio = 1.0):
    if T.shape[1] == 2:
        T = np.hstack((T,np.array([[0,0,1]]).T)).shape
    # update translation to current resize
    if translation_resize is None:
        translation_resize = image_resize/rotation_resize_ratio
    T[:2,-1] = T[:2,-1] * translation_resize
    h, w = img.shape[:2]
    # center and rotate
    center = np.array([[1, 0, w/2], [0, 1, h/2], [0, 0, 1]] )
    center_inverse = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]] )
    total_matrix = center @ T @ center_inverse
    total_affine = AffineTransform( matrix=total_matrix )
    img = warp( img, total_affine, output_shape= output_shape)
    return img

def read_hne_from_tif(file_name, grayscale = False, resize = 1.0, invert_color = False):
    hne_tiff = tifffile.imread(file_name)
    hne_np = np.transpose(hne_tiff, (1, 2, 0))/255
    del hne_tiff
    if resize != 1.0:
        hne_resized = cv2.resize(hne_np, 
                         (int(hne_np.shape[1] * resize), int(hne_np.shape[0] * resize)),
                         interpolation = cv2.INTER_AREA)
        del hne_np
    else:
        hne_resized = hne_np
    if grayscale: 
        hne_resized = skimage.color.rgb2gray(hne_resized)
    if invert_color:
        hne_resized = 1.0 - hne_resized
    return hne_resized

def read_cycif_channel_from_tif(file_name, resize = 1.0, channel = 0, log_transform = True, min_max_norm = True, quantile_norm = 0.0, rotate_90clockwise = True):
    cycif_tiff = tifffile.imread(file_name)
    if isinstance(channel, str):
        cycif_tiff_file = tifffile.TiffFile(file_name)
        root = xml.etree.ElementTree.fromstring(cycif_tiff_file.ome_metadata)
        cycif_channels = [e.attrib['Name'] for e in root.findall(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel")]
        channel = cycif_channels[channel]
    cycif_channel = cycif_tiff[channel]
    if resize != 1.0:
        cycif_resized = cv2.resize(cycif_channel, 
                                 (int(cycif_channel.shape[1] * resize), int(cycif_channel.shape[0] * resize)),
                                 interpolation = cv2.INTER_AREA)
    else:
        cycif_resized = cycif_channel.copy()
    del cycif_tiff, cycif_channel
    cycif_resized = np.array(cycif_resized, dtype=float)
    if log_transform:
        cycif_resized = np.log(cycif_resized+1)
    if quantile_norm>0:
        vals = cycif_resized.flatten()
        l = np.quantile(vals, quantile_norm)
        h = np.quantile(vals, 1-quantile_norm)
        cycif_resized[cycif_resized<=l] = l
        cycif_resized[cycif_resized>h] = h
        del vals
    if min_max_norm:
        cycif_resized = (cycif_resized - cycif_resized.min())/(cycif_resized.max() - cycif_resized.min())
        # cycif_resized = (np.array(cycif_resized, dtype=float) - 0)/(cycif_resized.max() -0)
    if rotate_90clockwise:
        cycif_resized = cv2.rotate(cycif_resized, cv2.ROTATE_90_CLOCKWISE)
    return cycif_resized


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

def sitk_affine_registration(fixed, moving, verbose = True):
    fixed = sitk.GetImageFromArray(fixed.astype(float))
    moving = sitk.GetImageFromArray(moving.astype(float))
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform()
    )
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    if verbose:
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx = R.Execute(fixed, moving)
    if verbose:
        print(outTx)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)
    return sitk.GetArrayFromImage(out), outTx

def sitk_BSpline_registration1(fixed, moving, verbose = True):
    fixed = sitk.GetImageFromArray(fixed.astype(float))
    moving = sitk.GetImageFromArray(moving.astype(float))
    
    transformDomainMeshSize = [8] * moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)
    if verbose:
        print(tx) 
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)
    if verbose:
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx = R.Execute(fixed, moving)
    
    if verbose:
        print(outTx)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)
    return sitk.GetArrayFromImage(out), outTx

def elastix_affine_registration(fixed, moving, verbose = True, NumberOfResolutions = None):
    fixedImage = sitk.GetImageFromArray(fixed)
    movingImage = sitk.GetImageFromArray(moving)

    parameterMap = sitk.GetDefaultParameterMap('affine')
    if NumberOfResolutions is not None:
        parameterMap['NumberOfResolutions'] = (str(NumberOfResolutions),)
    sitk.PrintParameterMap(parameterMap)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()
    
    return sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()), elastixImageFilter

def elastix_filter_to_homography(elastixImageFilter):
    l = elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters']
    l = [float(x) for x in l]
    T = np.array([[l[0], l[1], l[4]], [l[2], l[3], l[5]], [0, 0, 1]])
    return T

def sitk_composite_registration(fixed, moving, verbose = True, registration_functions = None):
    if registration_functions is None:
        registration_functions = [sitk_affine_registration, sitk_BSpline_registration1]
    imgs = []
    txs = []
    img = moving
    for func in registration_functions:
        img, tx = func(fixed, img, verbose = verbose)
        imgs.append(img)
        txs.append(tx)
    return imgs, txs

def global_affine_registration_elastix(fixed, moving, resize_ratio = 1.0, verbose = False, NumberOfResolutions = None):
    if resize_ratio != 1.0:
        fixed = resize_image_ratio(fixed, resize_ratio)
        moving = resize_image_ratio(moving, resize_ratio)
    _, elastixImageFilter = elastix_affine_registration(fixed, moving, verbose = verbose, NumberOfResolutions = NumberOfResolutions)
    return elastix_filter_to_homography(elastixImageFilter)

# def STalign_coordinates_to_image_alignment(image, W, rasterize_dx = 30, device = 'cuda:0', lddmm_params = None):
#     Inorm = STalign.normalize(image)
#     I = Inorm.transpose(2,0,1)
#     YI = np.array(range(I.shape[1]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
#     XI = np.array(range(I.shape[2]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
#     extentI = STalign.extent_from_x((YI,XI))
#     print('rasterize')
#     XJ,YJ,M,fig = STalign.rasterize(W[:,0],W[:,1], dx=rasterize_dx)
#     J = np.vstack((M, M, M)) # make into 3xNxM
#     J = STalign.normalize(J)
#     # run LDDMM
#     # keep all other parameters default
#     params = {
#           'niter':4000,#2000,
#           'device':device,
#           'sigmaM':0.15,
#           'sigmaB':0.10,
#           'sigmaA':0.11,
#           'epV': 10,
#           'muB': torch.tensor([0,0,0]).to(device), # black is background in target
#           'muA': torch.tensor([1,1,1]).to(device) # use white as artifact
#           }
#     if lddmm_params is not None:
#         for v in lddmm_params:
#             params[v] = lddmm_params[v]
#     print('LDDMM')
#     A,v,xv = STalign.LDDMM([YI,XI],I,[YJ,XJ],J,**params)
#     A = A.detach().cpu()
#     v = v.detach().cpu()
#     xv = [x.detach().cpu() for x in xv]
#     tpointsI = STalign.transform_points_target_to_atlas(xv,v,A,np.stack([W[:,1],W[:,0]], -1))
#     return tpointsI


def get_image_and_coordinates_patch(img, coordinates, x_start, y_start, values = None, patch_size = None, x_end = None, y_end = None, slack = 0, log_vals = False, plot = False, figsize=(15,15), s = 100):
    assert((patch_size is not None) or ((x_end is not None) and (y_end is not None)))
    if patch_size is not None:
        x_end = x_start + patch_size
        y_end = y_start + patch_size
    else:
        patch_size = np.mean(x_end-x_start, y_end - y_start)
    if (y_start>=0) and (x_start>=0) and (y_end<img.shape[0]) and (x_end<img.shape[1]):
        sub_image = img[y_start:y_end, x_start:x_end]
    else:
        sub_image = img[max(y_start,0):min(y_end,img.shape[0]), max(x_start,0):min(x_end,img.shape[1])]
        before_pad = lambda x: 0 if x>=0 else abs(x)
        after_pad = lambda x,y: 0 if x<=y else x-y
        sub_image = np.pad(sub_image, ((before_pad(y_start),after_pad(y_end, img.shape[0])),(before_pad(x_start),after_pad(x_end, img.shape[1]))),'constant', constant_values=0)
    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(sub_image, cmap='gray')
    where_points = np.logical_and(np.logical_and(coordinates[:,0]>=x_start-slack, coordinates[:,0]<x_end+slack),
                                  np.logical_and(coordinates[:,1]>=y_start-slack, coordinates[:,1]<y_end+slack))
    sub_coordinates = None
    if where_points.sum()>0:
        sub_coordinates = coordinates[where_points].copy()
        if values is None:
            sub_values = None
        else:
            sub_values = values[where_points] if not log_vals else np.log(1+values[where_points])
        sub_coordinates = sub_coordinates - np.array([x_start,y_start])
        if plot:
            plt.scatter(sub_coordinates[:,0], sub_coordinates[:,1], c=sub_values, cmap='viridis', s=s)
    else:
        return sub_image, where_points, np.array([[0,0]]), None
    if plot:
        plt.show()
    return sub_image, where_points, sub_coordinates, sub_values


def approxInversePoint(point, tx, eps = 1e-2, max_iter = 100, img_shape = None):
    # print('point', point)
    try:
        tx_inv = tx.GetInverse()
        return np.array(tx_inv.TransformPoint(point))
    except:
        pass
    delta = 2*eps
    new_point = np.array([0.0,0.0]) # point.copy()
    v = 0
    t = 0
    while delta>eps:
        new_point = new_point - v
        inverse_point = np.array(tx.TransformPoint(new_point))
        v = inverse_point - point
        delta = sum(v**2)
        t+=1
        if t >max_iter:
            break
    return new_point

def approxInversePoints(points, tx, eps = 1e-2, max_iter = 100):
    new_points = points.copy()
    for i in range(len(new_points)):
        new_points[i] = approxInversePoint(points[i], tx, eps = eps, max_iter = max_iter)
    return new_points

def approxInversePointsMultiTx(points, txs, eps = 1e-2, max_iter = 100):
    new_points = points.copy()
    for tx in txs:
        new_points = approxInversePoints(new_points, tx, eps = eps, max_iter= max_iter)
    return new_points


def extract_align_patch_coordinates(fixed_img, moving_img, coordinates, x_start, y_start, patch_size, slack = 0, plot = False, resize_registration = 1.0, min_spots = 0):
    fixed_sub_img, where_points, sub_coordinates, _ = get_image_and_coordinates_patch(fixed_img, coordinates, x_start, y_start, patch_size = patch_size, 
                                                                                               values = None, slack = 0, log_vals = False, 
                                                                                               plot = plot, figsize=(5,5), s = 2)
    moving_sub_img, where_points, sub_coordinates, _ = get_image_and_coordinates_patch(moving_img, coordinates, 
                                                                                                 x_start - slack, y_start - slack, patch_size = patch_size + 2*slack, 
                                                                                                values = None, slack = -2*slack, log_vals = False, 
                                                                                                plot = plot, figsize=(5,5), s = 2)
    
    if where_points.sum()<min_spots:
        return sub_coordinates + np.array([x_start, y_start]), where_points
    
    fixed_sub_img = resize_image_ratio(fixed_sub_img, resize_registration)
    moving_sub_img = resize_image_ratio(moving_sub_img, resize_registration)
    
    try:
        imgs, txs = sitk_composite_registration(fixed_sub_img, moving_sub_img, verbose = False, 
                                            registration_functions = [sitk_affine_registration, sitk_BSpline_registration1])
    except RuntimeError as e:
        print("Regisrtarion error:", e)
        return None, np.zeros(len(where_points))
    
    sub_coordinates_resized = sub_coordinates*resize_registration
    sub_coordinates_new = approxInversePointsMultiTx(sub_coordinates_resized, txs, eps = 1e-2, max_iter = 100)
    
    if plot:
        for img in [fixed_sub_img, imgs[1]]:
            plt.figure(figsize=(5,5))
            plt.imshow(img, cmap='gray')
            sns.scatterplot(x = sub_coordinates_new[:,0], y = sub_coordinates_new[:,1], s = 2, linewidth=0)
            plt.show()
            
    sub_coordinates_new_org_scale = sub_coordinates_new/resize_registration
    sub_coordinates_new_org_scale_shifted = sub_coordinates_new_org_scale + np.array([x_start, y_start])
    
    return sub_coordinates_new_org_scale_shifted, where_points

def correct_coordinates_with_patch_alignment(fixed_img, moving_img, coordinates, patch_size, stride = None, slack = 0, plot = False, resize_registration = 1.0, min_spots = 500):
    stride = stride or patch_size
    in_shape = np.array(fixed_img.shape)
    out_shape = np.array(np.floor((in_shape + patch_size + 2*slack)/stride)+1, dtype=int)
    print(out_shape)
    counts = np.zeros(len(coordinates))
    avg_coordinates = np.zeros(coordinates.shape)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            x_start = j*stride - patch_size//2
            y_start = i*stride - patch_size//2
            print(i,j,x_start,y_start)
            sub_coordinates , where_points = extract_align_patch_coordinates(fixed_img, moving_img, coordinates, 
                                                                             x_start = x_start, y_start = y_start, patch_size = patch_size, slack = slack, 
                                                                             plot = plot, resize_registration = resize_registration, min_spots = min_spots)
            print(where_points.sum())
            if where_points.sum()>=min_spots:
                counts += where_points
                avg_coordinates[where_points] += sub_coordinates
    avg_coordinates[counts == 0] = coordinates[counts == 0]
    counts = np.maximum(counts,1)
    avg_coordinates = avg_coordinates/counts.reshape((len(coordinates),1))
    return avg_coordinates, counts


def cells_per_patch(img, coordinates, patch_size, stride = None, slack = 0, iter_patches = False, plot = False, resize_registration = 1.0, min_spots = 1):
    stride = stride or patch_size
    in_shape = np.array(img.shape[:2])
    out_shape = np.array(np.floor((in_shape + patch_size + 2*slack)/stride)+1, dtype=int)
    out_shape_rev = out_shape[::-1]
    print(out_shape)
    cells_dict = {}
    if iter_patches:
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                x_start = j*stride - patch_size//2
                y_start = i*stride - patch_size//2
                x_end = x_start + patch_size
                y_end = y_start + patch_size
                # print(i, j, x_start, x_end, y_start, y_end)
                where_points = np.logical_and(np.logical_and(coordinates[:,0]>=x_start-slack, coordinates[:,0]<x_end+slack),
                                  np.logical_and(coordinates[:,1]>=y_start-slack, coordinates[:,1]<y_end+slack))
                if where_points.sum() < min_spots: continue
                sub_coordinates = coordinates[where_points] - np.array([x_start, y_start])
                which_points = list(np.where(where_points)[0])
                cells_dict[i, j] = which_points
                if plot:  
                    if (y_start>=0) and (x_start>=0) and (y_end<img.shape[0]) and (x_end<img.shape[1]):
                        sub_image = img[y_start:y_end, x_start:x_end]
                    else:
                        sub_image = img[max(y_start,0):min(y_end,img.shape[0]), max(x_start,0):min(x_end,img.shape[1])]
                        before_pad = lambda x: 0 if x>=0 else abs(x)
                        after_pad = lambda x,y: 0 if x<=y else x-y
                        sub_image = np.pad(sub_image, ((before_pad(y_start),after_pad(y_end, img.shape[0])),(before_pad(x_start),after_pad(x_end, img.shape[1])), (0,0)),'constant', constant_values=0)
                    plt.imshow(sub_image)
                    ax = sns.scatterplot(x = sub_coordinates[:,0], y = sub_coordinates[:,1], s = 1, linewidth=0 ,c = 'y')
                    plt.title(f"{i} {j}")
                    plt.show()
    else:
        max_ji = np.floor((coordinates + patch_size//2)/stride).astype(int) + 1
        min_ji = np.floor((coordinates + patch_size//2 - patch_size)/stride).astype(int) + 1
        min_ji  = np.maximum(min_ji, 0)
        max_ji = np.minimum(max_ji, out_shape_rev)
        for k in range(len(coordinates)):
            for j in range(min_ji[k,0], max_ji[k,0]):
                for i in range(min_ji[k,1], max_ji[k,1]):
                    d = cells_dict.get((i,j), [])
                    d.append(k)
                    cells_dict[i, j] = d
        if min_spots>1:
            cells_dict = dict(filter(lambda pair: len(pair[1])>=min_spots, cells_dict.items()))
    return cells_dict



# https://github.com/dovahcrow/patchify.py/blob/master/patchify/view_as_windows.py
def view_as_windows_arg_check(arr_in, window_shape, step=1, color_image = True):
    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")
    ndim = arr_in.ndim
    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim if not color_image else (window_shape, window_shape, 3)
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")
    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim if not color_image else (step, step, 3)
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")
    return window_shape, step
def view_as_windows(arr_in, window_shape, step=1, color_image = True):
    window_shape, step = view_as_windows_arg_check(arr_in, window_shape, step, color_image)
    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)
    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")
    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")
    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)
    indexing_strides = arr_in[slices].strides
    win_indices_shape = (
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))
    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out

def pad_image_after_for_patches(arr_in, window_shape, step, constant_values = 0,  color_image = True):
    window_shape, step = view_as_windows_arg_check(arr_in, window_shape, step=step, color_image = color_image)
    win_indices_shape_mod = ((np.array(arr_in.shape) - np.array(window_shape)) % np.array(step))
    pad_idx = np.array(step) - win_indices_shape_mod
    if color_image: pad_idx[-1] = 3
    padded_arr_in = np.pad(arr_in, ((0,pad_idx[0]),(0,pad_idx[1]), (0,0)),'constant', constant_values = constant_values)
    # plt.imshow(padded_arr_in)
    return padded_arr_in

def view_as_windows_with_padding(arr_in, window_shape, step=1, color_image = True, constant_values = 0, plot = False):
    patches = view_as_windows(arr_in, window_shape, step=step, color_image = color_image)
    window_shape, step = view_as_windows_arg_check(arr_in, window_shape, step=step, color_image = color_image)
    win_indices_shape_mod = ((np.array(arr_in.shape) - np.array(window_shape)) % np.array(step))
    print(patches.shape)
    if win_indices_shape_mod[0] != 0:
        sub_img = arr_in[patches.shape[0] * step[0]:,:]
        sub_img = np.pad(sub_img, ((0,window_shape[0]-sub_img.shape[0]),(0,0), (0,0)),'constant', constant_values = constant_values)
        patches_last_x = view_as_windows(sub_img, window_shape, step=step, color_image = color_image)
        patches = np.concatenate((patches, patches_last_x), axis=0)
    if win_indices_shape_mod[1] != 0:
        sub_img = arr_in[:,patches.shape[1] * step[1]:]
        x_pad = step[0] - win_indices_shape_mod[0] if win_indices_shape_mod[0] != 0 else 0
        sub_img = np.pad(sub_img, ((0,x_pad),(0,window_shape[1]-sub_img.shape[1]), (0,0)),'constant', constant_values = constant_values)
        patches_last_y = view_as_windows(sub_img, window_shape, step=step, color_image = color_image)
        patches = np.concatenate((patches, patches_last_y), axis=1)
    print(patches.shape)
    if plot:
        fig, axs = plt.subplots(patches.shape[0], patches.shape[1])
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                axs[i,j].imshow(patches[i,j,0])
                axs[i,j].axis('off')
        plt.show()
    
    return patches