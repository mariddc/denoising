#%% MODULES
import numpy as np
from skimage import io as skio
import heapq
import pywt
import math
from scipy.linalg import hadamard
import cv2
import matplotlib.pyplot as plt

#%% PARAMETERS
# Define parameters for the first step of the BM3D algorithm

# 1st step
kHard = 8  # Patch size
nHard = 39 # Search window size
NHard = 16 # Max number of similar patches kept
pHard = 3

sigma = 30
tauHard = 5000 if sigma > 40 else 2500

lambdaHard2d = 0  # Thresholding parameter for grouping
lambdaHard3d = 2.7

#%% GROUPING 1ST STEP
def get_search_window(image, x, y, patch_size=kHard, window_size=nHard):
    """ Gets the search window centered around the reference patch.
    
    Parameters:
    - image (np.ndarray): input image
    - x, y (int): coordinates of the top-left corner of the reference patch
    - patch_size (int): size of the patch
    - window_size (int): size of the search window
    
    Returns:
    - search_window (np.ndarray): search window around the reference patch
    - window_top_left_x, window_top_left_y (int): coordinates of the top-left corner of the search window
    """
    window_top_left_x = x - (window_size // 2 - patch_size // 2)
    window_top_left_y = y - (window_size // 2 - patch_size // 2)
    
    search_window = image[
        window_top_left_x: window_top_left_x + window_size,
        window_top_left_y: window_top_left_y + window_size 
    ]
    return search_window, window_top_left_x, window_top_left_y

def hard_thresholding(img, threshold):
    """ Applies hard thresholding to an array, setting values below the threshold to zero.
    
    Parameters:
    - img (np.ndarray): input array
    - threshold (float): threshold value
    
    Returns:
    - np.ndarray: thresholded array with values below the threshold set to zero
    """
    return (abs(img) > threshold) * img

def grouping_1st_step(x, y, image, sigma, patch_size, window_size, lambdaHard2d, tauHard, N=NHard):
    """ Groups similar patches around a reference patch within a search window.
    
    Parameters:
    - x, y (int): coordinates of the reference patch's top-left corner
    - image (np.ndarray): input image
    - sigma (float): noise standard deviation
    - patch_size (int): size of the patch
    - window_size (int): size of the search window
    - lambdaHard2d (float): thresholding parameter
    - tauHard (float): similarity threshold
    - N (int): max number of patches to keep
    
    Returns:
    - closer_patches (np.ndarray): selected patches similar to the reference
    - closer_coords (np.ndarray): coordinates of the selected patches
    """
    ref_patch = image[x:x+patch_size, y:y+patch_size]
    ref_patch_array = ref_patch.reshape(-1, patch_size**2)
    
    search_window, x_win, y_win = get_search_window(image, x, y, patch_size, window_size)
    window_patches = np.lib.stride_tricks.sliding_window_view(search_window, (patch_size, patch_size))
    window_patches_array = window_patches.reshape(-1, patch_size**2)
    
    if sigma > 40:
        ref_patch_array = hard_thresholding(ref_patch_array, lambdaHard2d * sigma)
        window_patches_array = hard_thresholding(window_patches_array, lambdaHard2d * sigma)

    diff_squared = (ref_patch_array - window_patches_array) **2
    ssd_array = np.sum(diff_squared, axis=1)
    dist_array = ssd_array / (kHard ** 2)
    
    N = 2 ** (math.floor(math.log2(N)))
    closer_indeces = dist_array.argsort()[:N]
    closer_indeces = np.array([i for i in closer_indeces if dist_array[i] < tauHard])
    
    size = len(closer_indeces)
    if not (size & (size-1) == 0):
        new_size = 2 ** (math.floor(math.log2(size)))
        closer_indeces = closer_indeces[:new_size]
    
    closer_coords = np.array([[x_win+(i//(window_size - patch_size + 1)), y_win+(i%(window_size - patch_size + 1))] for i in closer_indeces])
    closer_patches = np.array([window_patches_array[i].reshape(patch_size, patch_size) for i in closer_indeces])

    return closer_patches, closer_coords

#%% COLLABORATIVE FILTERING
def walsh_hadamard_transform(x):
    """ Applies the Walsh-Hadamard transform to a 1D array.
    
    Parameters:
    - x (np.ndarray): 1D input array of length 2^n, where n is a non-negative integer.
    
    Returns:
    - np.ndarray: transformed array
    """
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError("Length of input array must be a power of 2")

    if n == 1:
        return x
    
    even = walsh_hadamard_transform(x[0::2])
    odd = walsh_hadamard_transform(x[1::2])

    return np.concatenate([even + odd, even - odd]) 

def get_Bior_matrices(N=8):
    """ Computes Biorthogonal wavelet transform matrices.
    
    Parameters:
    - N (int): size of the matrix
    
    Returns:
    - directBior15_matrix, invBior15_matrix (np.ndarray): forward and inverse matrices
    """
    directBior15_matrix = np.zeros((N, N))
    ss = N // 2
    ls = []

    while ss > 0:
        ls.append(ss)
        ss = ss // 2
    for k in range(N):
        inp = np.zeros(N)
        inp[k] = 1
        tmp = inp
        out = []
        for s in ls:
            (a, b) = pywt.dwt(tmp, 'bior1.5', mode='periodic')
            out = list(b[0:s]) + out
            tmp = a[:s]
        out = list(a[:s]) + out
        directBior15_matrix[k, :] = np.asarray(out)

    invBior15_matrix = np.linalg.inv(directBior15_matrix)
    return directBior15_matrix, invBior15_matrix

def apply_bior(V, M, dim):
    """ Applies Biorthogonal transformation along a given dimension.
    
    Parameters:
    - V (np.ndarray): input array
    - M (np.ndarray): transformation matrix
    - dim (int): dimension along which to apply the transformation
    
    Returns:
    - np.ndarray: transformed array
    """
    s = V.shape
    l = [0, 1, 2]
    l[dim] = 0
    l[0] = dim
    smod = list(s)
    smod[dim] = s[0]
    smod[0] = s[dim]
    return (M @ V.transpose(l).reshape((M.shape[0], -1))).reshape(smod).transpose(l)

def apply_1d_transform(array, use_dct=False):
    """ Applies a 1D transform (DCT or Walsh-Hadamard) to an array.
    
    Parameters:
    - array (np.ndarray): input array
    - use_dct (bool): whether to use DCT instead of Walsh-Hadamard
    
    Returns:
    - np.ndarray: transformed array
    """
    return walsh_hadamard_transform(array) / np.sqrt(len(array))

B8, IB8 = get_Bior_matrices(N=kHard)

def apply_2d_transform(v, use_dct=False):
    """ Applies a 2D transform (DCT or Biorthogonal wavelet) to an array.
    
    Parameters:
    - v (np.ndarray): input array
    - use_dct (bool): whether to use DCT
    
    Returns:
    - np.ndarray: transformed array
    """
    if use_dct:
        return cv2.dct(v)
    else:
        v1d = apply_bior(v, B8, -1)
        v2d = apply_bior(v1d, B8, -2)
        return v2d

def reverse_2d_transform(v, use_dct=False):
    """ Reverses a 2D transform (DCT or Biorthogonal wavelet).
    
    Parameters:
    - v (np.ndarray): input array
    - use_dct (bool): whether to use DCT
    
    Returns:
    - np.ndarray: inversely transformed array
    """
    if use_dct:
        return cv2.dct(v, flags=cv2.DCT_INVERSE)
    else:
        vappinv1d = apply_bior(v, IB8, -1)
        vappinv2d = apply_bior(vappinv1d, IB8, -2)
        return vappinv2d

#%% AGGREGATION
def update_aggregation_buffers(nu, delta, patches, coords, weights, X, Y):
    """ Updates the aggregation buffers with weighted patches.
    
    Parameters:
    - nu (np.ndarray): numerator aggregation buffer
    - delta (np.ndarray): denominator aggregation buffer
    - patches (np.ndarray): filtered patches
    - coords (np.ndarray): coordinates of patches
    - weights (np.ndarray): weights for each patch
    - X, Y (np.ndarray): meshgrid arrays for patch positions
    
    Returns:
    - nu, delta (np.ndarray): updated aggregation buffers
    """
    weights = weights.reshape(-1,1,1)
    
    # Offset the grid for each patch position
    X_ = X + coords[:, 0].reshape(-1, 1, 1)  
    Y_ = Y + coords[:, 1].reshape(-1, 1, 1)  
    
    np.add.at(nu, (X_, Y_), patches * weights)
    np.add.at(delta, (X_, Y_), weights) 
    
    return nu, delta

#%% 1ST STEP
def bm3d_1st_step(image, sigma, kHard, nHard, lambdaHard2d, lambdaHard3d, tauHard, NHard):
    """ Executes the first step of the BM3D algorithm.
    
    Parameters:
    - image (np.ndarray): noisy input image
    - sigma (float): noise standard deviation
    - kHard (int): patch size
    - nHard (int): search window size
    - lambdaHard2d, lambdaHard3d (float): thresholding parameters
    - tauHard (float): similarity threshold
    - NHard (int): max number of patches to keep
    
    Returns:
    - basic (np.ndarray): denoised image after the first step
    """
    height, width = image.shape

    # pad image and iterate through original frame
    window_size = nHard
    offset = window_size // 2
    padded_image = np.pad(image, offset, mode='reflect')

    nu = np.zeros(padded_image.shape)
    delta = np.zeros(padded_image.shape)

    X, Y = np.meshgrid(np.arange(kHard), np.arange(kHard), indexing='ij')
    
    # iterate through patches in the image with a step
    for x in range(offset, offset + height - kHard + 1, pHard):
        for y in range(offset , offset + width - kHard + 1, pHard):

            # GROUPING
            group3d, coords = grouping_1st_step(x, y, padded_image, sigma, patch_size=kHard, window_size=nHard, lambdaHard2d=lambdaHard2d, tauHard=tauHard, N=NHard)
            if len(coords) < 1:
                continue

            # COLLABORATIVE FILTERING
            # 3d transform
            transformed = np.array(apply_2d_transform(group3d, use_dct=False))
            transformed = apply_1d_transform(transformed) #t1d

            # thresholding
            threshold = lambdaHard3d * sigma
            thresholded = hard_thresholding(transformed, threshold)

            # calculate weights
            NPHard = np.count_nonzero(thresholded)
            weight = 1 / NPHard if NPHard >= 1 else 1
            weight = np.array([weight])

            # 3d reverse
            thresholded = apply_1d_transform(thresholded) #t1d
            filtered = np.array(reverse_2d_transform(thresholded, use_dct=False))

            # AGGREGATION 
            nu, delta = update_aggregation_buffers(nu, delta, filtered, coords, weight, X, Y)

    # Compute the basic estimate by dividing aggregated values
    basic = np.divide(nu[offset:offset+height, offset:offset+width], delta[offset:offset+height, offset:offset+width])

    return basic

#%% INITIALIZATION
def noise(im, br):
    """ Adds white Gaussian noise to the image with a specified standard deviation.
    
    Parameters:
    - im (np.ndarray): original image
    - br (float): standard deviation of noise
    
    Returns:
    - np.ndarray: noisy image
    """
    imt = np.float32(im.copy())
    bruit = br * np.random.randn(*imt.shape)
    return imt + bruit

# Read and display the images
im = skio.imread('./lena.tif') # original image
imbr = noise(im, sigma) # create noisy image

# Display original, noisy, and denoised images
print("Original Image:")
plt.title("Original Image")
plt.imshow(im, cmap='gray')
plt.show()

print("Image with noise:")
plt.title("Image with noise")
plt.imshow(imbr, cmap='gray')
plt.show()

basic_estimate = bm3d_1st_step(imbr, sigma, kHard, nHard, lambdaHard2d, lambdaHard3d, tauHard, NHard)

print("Basic estimate after 1st step:")
plt.title("Basic estimate after 1st step")
plt.imshow(basic_estimate, cmap='gray')
plt.show()
