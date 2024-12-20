#%% MODULES
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import math, time, os

#%% PARAMETERS
# 1st step
kHard = 8  # Patch size
nHard = 39 # Search window size
NHard = 16 # Max number of similar patches kept
pHard = 3
lambdaHard2d = 0  # Thresholding parameter for grouping
lambdaHard3d = 2.7

# 2nd step
kWien = 8
nWien = 39
NWien = 32
pWien = 3


def define_thresholds(sigma):
    tauHard = 5000 if sigma > 40 else 2500
    tauWien = 3500 if sigma > 40 else 400
    return tauHard, tauWien

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
    window_top_left_x = x - (window_size//2 - patch_size//2)
    window_top_left_y = y - (window_size//2 - patch_size//2)
    
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
    # reference patch as array
    ref_patch = image[x:x+patch_size, y:y+patch_size]
    ref_patch_array = ref_patch.reshape(-1, patch_size**2)
    
    # vectorized patches from search window
    search_window, x_win, y_win = get_search_window(image, x, y, patch_size, window_size)
    window_patches = np.lib.stride_tricks.sliding_window_view(search_window, (patch_size, patch_size))
    window_patches_array = window_patches.reshape(-1, patch_size**2)
    
    # hard thresholding
    if sigma > 40:
        ref_patch_array = hard_thresholding(ref_patch_array, lambdaHard2d * sigma)
        window_patches_array = hard_thresholding(window_patches_array, lambdaHard2d * sigma)

    # calculate vector differences to reference patch
    diff_squared = (ref_patch_array - window_patches_array) **2
    ssd_array = np.sum(diff_squared, axis=1)
    dist_array = ssd_array / (kHard ** 2)
    
    # get N closest patches, N must be power of 2 and distance must be < tauHard
    N = 2 ** (math.floor(math.log2(N)))
    closer_indeces = dist_array.argsort()[:N]
    closer_indeces = np.array([i for i in closer_indeces if dist_array[i] < tauHard]) #apply similarity threshold
    
    size = len(closer_indeces)
    if not (size & (size-1) == 0):
        new_size = 2 ** (math.floor(math.log2(size)))
        closer_indeces = closer_indeces[:new_size]
    
    # get top left coord of each patch and build the 3d group
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

def apply_1d_transform(array):
    """ Applies a 1D transform (Walsh-Hadamard) to an array.
    
    Parameters:
    - array (np.ndarray): input array
    
    Returns:
    - np.ndarray: transformed array normalized
    """
    return walsh_hadamard_transform(array) / np.sqrt(len(array))

B8, IB8 = get_Bior_matrices(N=kHard)
def apply_2d_transform(v, use_dct=False):
    """Applies a 2D transform (either DCT or Biorthogonal) on a set of blocks or coefficients.
    
    Parameters:
    - v (np.ndarray): A collection of 2D arrays (3d group).
    - use_dct (bool): If True, use the DCT-based 2D transform. Otherwise, use the Biorthogonal transform.
    
    Returns:
    - np.ndarray: The transformed data.
    """
    if use_dct:
        return dct(dct(v, axis=1, norm='ortho'), axis=2, norm='ortho')
    else:
        v1d=apply_bior(v,B8,-1)
        v2d=apply_bior(v1d,B8,-2)
        return v2d

def reverse_2d_transform(v, use_dct=False):
    """Reverses a 2D transform (either DCT or Biorthogonal) on a set of blocks or coefficients.
    
    Parameters:
    - v (np.ndarray): A collection of 2D transformed blocks.
    - use_dct (bool): If True, use the DCT-based 2D inverse transform. Otherwise, use the inverse Biorthogonal transform.
    
    Returns:
    - numpy.ndarray: The reconstructed (inverse-transformed) data.
    """
    if use_dct:
        return idct(idct(v, axis=2, norm='ortho'), axis=1, norm='ortho') 
    else:
        vappinv1d=apply_bior(v,IB8,-1)
        vappinv2d=apply_bior(vappinv1d,IB8,-2)
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
    for x in range(offset, offset + height - kHard + pHard, pHard):
        for y in range(offset , offset + width - kHard + pHard, pHard):

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
#%% SECOND STEP
def grouping_2nd_step(x, y, image, basic, sigma, patch_size, window_size, lambdaHard2d, tauWien, NWien):
    """
    Forms groups of patches for the second step of the BM3D algorithm.

    - x, y (int): Coordinates of the top-left corner of the reference patch.

    Returns:
    - original_patches (np.ndarray): 3D array of patches grouped from the original noisy image.
    - basic_patches (np.ndarray): 3D array of patches grouped from the basic estimate.
    - basic_coords (list of tuples): List of coordinates of the grouped patches.
    """
    #group formed by the basic estimate
    basic_patches, basic_coords = grouping_1st_step(x, y, basic, sigma, patch_size, window_size, lambdaHard2d, tauWien, NWien)

    #group formed by the original image
    original_patches = np.array([image[i:i+patch_size, j:j+patch_size] for i,j in basic_coords])

    return original_patches, basic_patches, basic_coords

def bm3d_2nd_step(image, basic_estimate, sigma, kWien, nWien, lambdaHard2d, lambdaHard3d, tauWien, NWien):
    """
    Executes the second step of the BM3D algorithm.

    Parameters:
    - image (np.ndarray): Original noisy input image.
    - basic_estimate (np.ndarray): Basic estimate of the denoised image from the first step.
    - sigma (float): Noise standard deviation.
    - kWien (int): Patch size.
    - nWien (int): Search window size.
    - lambdaHard2d (float): Thresholding parameter for patch similarity.
    - lambdaHard3d (float): Thresholding parameter for 3D filtering.
    - tauWien (float): Similarity threshold for Wiener grouping.
    - NWien (int): Maximum number of patches to group.

    Returns:
    - final (np.ndarray): Final denoised image after the second step.
    """
    height, width = image.shape

    # pad image and iterate through original frame
    window_size = nWien
    offset = window_size // 2
    padded_image = np.pad(image, offset, mode='reflect')
    padded_basic = np.pad(basic_estimate, offset, mode='reflect')

    nu = np.zeros(padded_image.shape)
    delta = np.zeros(padded_image.shape)

    X, Y = np.meshgrid(np.arange(kWien), np.arange(kWien), indexing='ij')
    
    # iterate through patches in the image with a step
    for x in range(offset, offset + height - kWien + pWien, pWien):
        for y in range(offset , offset + width - kWien + pWien, pWien):

            # GROUPING
            group3d_original, group3d_basic, coords = grouping_2nd_step(x, y, padded_image, padded_basic, sigma, patch_size=kWien, window_size=nWien, lambdaHard2d=lambdaHard2d, tauWien=tauWien, NWien=NWien)
            if len(coords) < 1:
                continue

            # COLLABORATIVE FILTERING
            # basic 3D transform
            basic_transformed = np.array(apply_2d_transform(group3d_basic, use_dct=True))
            basic_transformed = apply_1d_transform(basic_transformed) 

            # wiener coefficients over basic estimation
            module = np.absolute(basic_transformed) ** 2
            wp = module / (module + sigma**2) 

            # original 3D transform
            original_transformed = np.array(apply_2d_transform(group3d_original, use_dct=True))
            original_transformed = apply_1d_transform(original_transformed) 

            # wiener filtering over original
            original_filtered = wp * original_transformed

            # 3D reverse
            filtered = apply_1d_transform(original_filtered)
            filtered = np.array(reverse_2d_transform(filtered, use_dct=True))

            # calculate weights
            weight = np.array([np.linalg.norm(wp) ** (-2)])

            # AGGREGATION 
            nu, delta = update_aggregation_buffers(nu, delta, filtered, coords, weight, X, Y)

    ## FINAL ESTIMATE
    final = np.divide(nu[offset:offset+height, offset:offset+width], delta[offset:offset+height, offset:offset+width])

    return final


#%% EVALUATION
def compute_rmse(reference_image, denoised_image):
    """
    Compute the Root Mean Square Error (RMSE) between two images.
    
    Parameters:
        reference_image (numpy.ndarray): The noiseless reference image (u_R).
        denoised_image (numpy.ndarray): The denoised image (u_D).
        
    Returns:
        float: The RMSE value.
    """ 
    # Compute RMSE using vectorized operations
    mse = np.mean((reference_image - denoised_image) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if the images are identical
    rmse = np.sqrt(mse)
    
    return rmse

def compute_psnr(reference_image, denoised_image):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters:
        reference_image (numpy.ndarray): The noiseless reference image.
        denoised_image (numpy.ndarray): The denoised image.
        
    Returns:
        float: The PSNR value in decibels (dB).
    """
    # Ensure the two images have the same shape
    assert reference_image.shape == denoised_image.shape, "Images must have the same dimensions"
    
    rmse = compute_rmse(reference_image, denoised_image)
    max_pixel_value = 255.0  # 8-bit images with pixel values in [0, 255]
    psnr = 20 * np.log10(max_pixel_value / rmse)
    
    return psnr

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
    imt += bruit
    return imt

def normalize (image, vmin=0, vmax=255):
    """
    Normalizes image values to a specified range
    """
    image = (image - image.min()) / (image.max() - image.min()) * vmax
    image = np.clip(image, vmin, vmax).astype(np.float32)
    return image

#%% PLOT RESULTS
def plot_results(original, noisy, basic, final, sigma, psnr1, psnr2, ex_time, figsize=(10, 8)):
    images = [original, noisy, basic, final]
    titles = ['Original image', f'Noisy image (std dev = {sigma})', 'Basic estimate (1st step)', 'Final estimate(2nd step)']
    subtitles = [f'PSNR: {psnr1:.2f}', f'PSNR: {psnr2:.2f}']
    note = f'Execution time: {ex_time:.2f}s'

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray') 
        ax.set_title(titles[i], fontsize=12, pad=10) 
        ax.axis('off') 
        if i >= 2: 
            ax.text(
                0.5, -0.05, subtitles[i - 2],  
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='black'
            )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    fig.text(0.95, 0.03, note, ha='right', fontsize=8)
    return fig

#%% EXAMPLE OF USAGE

def run_bm3d(noisy_image, sigma):
    """Runs the full BM3D denoising algorithm
    Parameters:
    - Noisy image (np.ndarray): image to be denoised
    - Sigma (int): standard deviation of the gaussian noise applied
    Returns:
    - final_estimate (np.ndarray): denoised image
    """
    tauHard, tauWien = define_thresholds(sigma)
    basic_estimate = bm3d_1st_step(imbr, sigma, kHard, nHard, lambdaHard2d, lambdaHard3d, tauHard, NHard)
    final_estimate = bm3d_2nd_step(imbr, basic_estimate, sigma, kWien, nWien, lambdaHard2d, lambdaHard3d, tauWien, NWien)

    return final_estimate

def run_bm3d_and_save(image_file, original_image, noisy_image, sigma):
    tauHard, tauWien = define_thresholds(sigma)

    #prepare
    im = original_image
    imbr = noisy_image
    
    #denoise
    start = time.time()
    basic_estimate = bm3d_1st_step(imbr, sigma, kHard, nHard, lambdaHard2d, lambdaHard3d, tauHard, NHard)
    final_estimate = bm3d_2nd_step(imbr, basic_estimate, sigma, kWien, nWien, lambdaHard2d, lambdaHard3d, tauWien, NWien)
    end = time.time()

    #evaluate
    basic_psnr = compute_psnr(im, basic_estimate)
    final_psnr = compute_psnr(im, final_estimate)
    exec_time = end - start
    #print(exec_time)
    
    basic_estimate = np.clip(basic_estimate, a_min=0.0, a_max=255.0)
    final_estimate = np.clip(final_estimate, a_min=0.0, a_max=255.0)
    imbr = np.clip(imbr, a_min=0.0, a_max=255.0)
    
    #plot
    fig = plot_results(original=im, noisy=imbr, basic=basic_estimate, final=final_estimate, sigma=sigma, psnr1=basic_psnr, psnr2=final_psnr, ex_time=exec_time)
    
    image_name, _ = os.path.splitext(image_file)
    plt.savefig(f'./bm3d-results-plt/{image_name}s{sigma}-plot.png', dpi=300, bbox_inches='tight')
    plt.imsave(f'./bm3d-results/{image_name}s{sigma}-noisy.jpg', imbr, cmap='gray') 
    plt.imsave(f'./bm3d-results/{image_name}s{sigma}-basic.jpg', basic_estimate, cmap='gray') 
    plt.imsave(f'./bm3d-results/{image_name}s{sigma}-denoised.jpg', final_estimate, cmap='gray') 
    plt.close()

    return final_estimate, final_psnr



if __name__ == '__main__':
    # inputs
    image_file = 'lena.tif'
    sigma = 30

    im = skio.imread(f'./images/{image_file}', as_gray=True) 
    im = normalize(im)
    imbr = noise(im, sigma) # create noisy image

    start = time.time()
    denoised = run_bm3d(imbr, sigma)
    end = time.time()

    print(f'PSNR: {compute_psnr(im, denoised):.2f}')

    print(f'Execution time: {end - start: .2f}s')
    
    denoised = np.clip(denoised, a_min=0.0, a_max=255.0)
    plt.imshow(denoised, cmap='gray')
    plt.show()

    #run_bm3d_and_save(image_file, im, imbr, sigma)



