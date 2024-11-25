import numpy as np
from skimage import io as skio
import math
import time, os

import matplotlib.pyplot as plt

#%% PARAMETERS
def define_parameters(sigma):
    """
    Defines parameters for the denoising algorithm based on the noise standard deviation (sigma).
    The values follow the table in the IPOL article.
    """
    match sigma:
        case sigma if sigma in range(0,16):
            patch_size = 3
            window_size = 21
            h = 0.4 * sigma
        case sigma if sigma in range(16,31):
            patch_size = 5
            window_size = 21
            h = 0.4 * sigma
        case sigma if sigma in range(31,46):
            patch_size = 7
            window_size = 35
            h = 0.35 * sigma
        case sigma if sigma in range(46,76):
            patch_size = 9
            window_size = 35
            h = 0.35 * sigma
        case sigma if sigma in range(76,101):
            patch_size = 11
            window_size = 35
            h = 0.3 * sigma
        case _:
            print("sigma must be between 0 and 100")
    
    patch_f  =  (patch_size - 1) // 2
    window_r = (window_size - 1) // 2
    
    return patch_f, window_r, h

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

def normalize(image, vmin=0.0, vmax=255.0):
    """
    Normalizes image values to a specified range
    """
    image = (image - image.min()) / (image.max() - image.min()) * vmax
    image = np.clip(image, vmin, vmax).astype(np.float32)
    return image

#%% NON-LOCAL MEANS
def calculate_weights(dist_array, sigma, h):
    """
    Calculates the weights for each patch using the Non-Local Means formula.
    Similar patches have more weight.
    """
    array = dist_array - (2 * (sigma ** 2))
    exponents = - np.maximum(array, 0.0) / (h**2)

    return np.exp(exponents)

def nlmeans(image, sigma):
    """
    Applies Non-Local Means denoising to the input image with noise of standard deviation sigma.
    """
    height, width = image.shape

    f, r, h = define_parameters(sigma)
    patch_size = 2 * f + 1
    window_size = 2 * r + 1 

    #for border handling
    offset = window_size
    padded_image = np.pad(image, offset, mode='reflect')

    denoised = np.zeros(padded_image.shape)

    for x in range(offset-f, offset + height+f+1):
        for y in range(offset-f , offset + width+f+1):
            # flattens reference patch
            ref_patch = padded_image[x-f:x+f+1, y-f:y+f+1]        
            ref_patch_array = ref_patch.reshape(-1, patch_size**2)

            # flattens all patches within the search window
            search_window = padded_image[x-r:x+r+1, y-r:y+r+1]
            window_patches = np.lib.stride_tricks.sliding_window_view(search_window, (patch_size, patch_size))
            window_patches_array = window_patches.reshape(-1, patch_size**2)

            # calculate vector differences to reference patch
            diff_squared = (ref_patch_array - window_patches_array) **2
            ssd_array = np.sum(diff_squared, axis=1)
            dist_array = ssd_array / (patch_size ** 2)

            # weighted aggregation
            weights = calculate_weights(dist_array, sigma, h)
            total_weight = np.sum(weights)
            weighed_sum = weights.reshape(-1,1) * window_patches_array 
            normalized = np.sum(weighed_sum, axis=0) / total_weight

            # update denoised image
            denoised[x-f:x+f+1, y-f:y+f+1] += normalized.reshape(patch_size, patch_size)
    # normalize result and crop the image back to original size
    denoised = denoised / (patch_size**2)
    denoised = denoised[offset:offset+height, offset:offset+width]

    return denoised
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
    assert reference_image.shape == denoised_image.shape, "Images must have the same dimensions"

    rmse = compute_rmse(reference_image, denoised_image)
    max_pixel_value = 255.0 # 8-bit images with pixel values in [0, 255]
    psnr = 20 * np.log10(max_pixel_value / rmse)
    
    return psnr

#%% PLOT RESULTS
def plot_results(original_image, noisy_image, denoised_image, sigma, psnr_value, execution_time):
    fig, axs = plt.subplots(1, 3, figsize=(12,6))

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(noisy_image, cmap='gray')
    axs[1].set_title(f"Noisy Image (std dev = {sigma})")
    axs[1].axis('off')

    axs[2].imshow(denoised_image, cmap='gray')
    axs[2].set_title(f"Denoised Image (PSNR: {psnr_value:.2f})")
    axs[2].axis('off')

    plt.subplots_adjust(bottom=0.2)
    note = f'Execution time: {execution_time:.2f}s'
    fig.text(0.95, 0.03, note, ha='right', fontsize=8)
    plt.tight_layout() 
    return fig

def run_nlm_and_save(image_file, original_image, noisy_image, sigma):
    #denoise
    start = time.time()
    denoised = nlmeans(noisy_image, sigma)
    end = time.time()

    #evaluate
    psnr = compute_psnr(original_image, denoised)
    exec_time = end - start

    original_image = np.clip(original_image, a_min=0.0, a_max=255.0)
    noisy_image = np.clip(noisy_image, a_min=0.0, a_max=255.0)
    denoised = np.clip(denoised, a_min=0.0, a_max=255.0)

    #plot
    fig = plot_results(original_image, noisy_image, denoised, sigma, psnr, exec_time)
    
    image_name, _ = os.path.splitext(image_file)
    plt.savefig(f'./nlm-results-plt/{image_name}s{sigma}-plot.png', dpi=300, bbox_inches='tight')
    plt.imsave(f'./nlm-results/{image_name}s{sigma}-noisy.jpg', noisy_image, cmap='gray') 
    plt.imsave(f'./nlm-results/{image_name}s{sigma}-denoised.jpg', denoised, cmap='gray') 

    plt.close()

    return denoised, psnr


#%% EXAMPLE OF USAGE

if __name__ == '__main__':
    # inputs
    image_file = 'muro.tif'
    sigma = 40

    im = skio.imread(f'./images/{image_file}', as_gray=True)
    im = normalize(im)
    imbr = noise(im, sigma) # create noisy image

    start = time.time()
    denoised = nlmeans(imbr, sigma)
    end = time.time()

    print(f'Execution time: {end - start: .2f}s')
    print(compute_psnr(im, denoised))

    denoised = np.clip(denoised, a_min=0.0, a_max=255.0)
    plt.imshow(denoised, cmap='gray')
    plt.show()

    #run_nlm_and_save(image_file, im, imbr, sigma)

