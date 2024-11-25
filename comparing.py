import bm3d
import nlmeans
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio


def plot_comparison(original, noisy, sigma, denoised_bm3d, denoised_nlm, psnr_bm3d, psnr_nlm):
    diff_bm3d = np.clip(np.abs(original - denoised_bm3d), a_min=0.0, a_max=255.0)
    diff_nlm = np.clip(np.abs(original - denoised_nlm), a_min=0.0, a_max=255.0)
    original = np.clip(original, a_min=0.0, a_max=255.0)
    noisy = np.clip(noisy, a_min=0.0, a_max=255.0)
    denoised_bm3d = np.clip(denoised_bm3d, a_min=0.0, a_max=255.0)
    denoised_nlm = np.clip(denoised_nlm, a_min=0.0, a_max=255.0)

    images = [original, denoised_bm3d, denoised_nlm, noisy, diff_bm3d, diff_nlm]
    titles = ['Original', f'BM3D (psnr={psnr_bm3d:.2f})', f'NLMeans (psnr={psnr_nlm:.2f})', f'Noisy (std dev={sigma})', 'Difference (Original - BM3D)', 'Difference (Original - NLM)']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Comparing both methods", fontsize=16, weight='bold') 

    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig


image_files = ['lena.tif', 'wall.tif', 'grille.tif', 'moon.tif', 'aerial.tif', 'facade.tif', 'harbor.tif', 'pepper.tif', 'chess.tif']
sigmas = [15, 30, 40, 60]

total = len(image_files) * len(sigmas)
i=1

start = time.time()
for image_file in image_files:
    im = skio.imread(f'./images/{image_file}', as_gray=True) 
    im = bm3d.normalize(im)

    for sigma in sigmas:
        imbr = bm3d.noise(im, sigma) # create noisy image

        print(f'For {image_file} and sigma = {sigma}:')
        print('  Starting BM3D...')
        denoised_bm3d, psnr_bm3d = bm3d.run_bm3d_and_save(image_file, im, imbr, sigma)
        print(f'  {i}/{total} BM3D done at {time.time()-start:.2f}.')
        print('  Starting NLMeans...')
        denoised_nlm, psnr_nlm  = nlmeans.run_nlm_and_save(image_file, im, imbr, sigma)        
        print(f'  {i}/{total} NLMeans done at {time.time()-start:.2f}.')

        fig = plot_comparison(im, imbr, sigma, denoised_bm3d, denoised_nlm, psnr_bm3d, psnr_nlm)
        
        image_name, _ = os.path.splitext(image_file)
        save_path = f'./comparing-results/{image_name}s{sigma}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'  Comparison saved at: {save_path}')
        
        plt.close()
        i+=1

end = time.time()
print(f'Tests execution duration: {end - start:.2f}s')
