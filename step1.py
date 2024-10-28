# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:13:45 2024

@author: maria
"""
#%% 
import numpy as np
from skimage import io as skio
import heapq
import pywt
import math
from scipy.linalg import hadamard
import cv2

# local
import matplotlib.pyplot as plt
#%% PARAMETERS
# 1st step
kHard = 8 #patch size
nHard = 39 #search window size --! era pra ser 39 mas nao entendi como centralizar P
NHard = 16 #max number of similar patches kept 
pHard = 3

sigma = 30
tauHard = 5000 if sigma > 40 else 2500

lambdaHard2d = 0 #hard thresholding for grouping --! ??? where
lambdaHard3d = 2.7

#%% INITIALIZATION
def noise(im,br):
    """ Cette fonction ajoute un bruit blanc gaussier d'ecart type br
       a l'image im et renvoie le resultat"""
    imt=np.float32(im.copy())
    sh=imt.shape
    bruit=br*np.random.randn(*sh)
    imt=imt+bruit
    return imt


#%% GROUPING
def closest_power_of_two(n, max_n):
    """Find the closest power of 2 to the number n, but not exceeding max_n."""
    if n == 0:
        return 0
    closest_pow2 = 2 ** (math.floor(math.log2(n)))
    return min(closest_pow2, max_n)


def distance(p,q):
    return (np.linalg.norm(p-q) ** 2) / (kHard ** 2)

def hard_thresholding(img, threshold):
    return (abs(img) <= threshold) * img

# x,y is the top-left corner of the reference patch
# doesnt work well for even window_size --change
def get_search_window(image, x, y, patch_size=kHard, window_size=nHard):
    img_h, img_w = image.shape  # image dimensions
    
    # padded image (to handle borders)
    padded_image = np.pad(image, window_size//2, mode='reflect')
    
    # adjust coordinates
    x_padded = x + window_size//2
    y_padded = y + window_size//2

    # ensure the patch defined by (x, y) fits within the image bounds
    if x < 0 or y < 0 or x + patch_size > img_w or y + patch_size > img_h:
        raise ValueError("The specified patch defined by (x, y) exceeds image boundaries.")
    
    search_window = padded_image[
        y_padded - (window_size//2 - patch_size//2):y_padded + window_size//2 + patch_size//2 +1,
        x_padded - (window_size//2 - patch_size//2):x_padded + window_size//2 + patch_size//2 +1  
    ]
    return search_window


def build_3d_group(p, window, sigma, lambdaHard2d, tauHard, N=NHard):
    closer_N_dists = []

    # assumming square patch and window
    k = p.shape[0]
    n = window.shape[0] 

    if sigma > 40:
        p = hard_thresholding(p, lambdaHard2d * sigma)
    
    for i in range(n-k+1):
        for j in range(n-k+1):
            # get patch Q and calculate distance to ref P
            q = window[i:k+i, j:k+j]
            if sigma > 40:
                q = hard_thresholding(q, lambdaHard2d * sigma)
            
            dist = distance(p, q)
            if dist <= tauHard:
                dist_tuple = (-dist, (i, j))  # negate distance to use max-heap
        
                if len(closer_N_dists) < N: # because after we will take out the first one
                    heapq.heappush(closer_N_dists, dist_tuple)
                else:
                    if dist_tuple > closer_N_dists[0]:
                        heapq.heappushpop(closer_N_dists, dist_tuple)
                        
    closer_N_dists = [(-d, idx) for d, idx in closer_N_dists]
    closer_N_dists = sorted(closer_N_dists, key=lambda x: x[0])

    
    group_3d = []
    for _, (i, j) in closer_N_dists:
        patch = window[i:k+i, j:k+j]
        group_3d.append(patch)
    group_3d=np.array(group_3d)
    
    return group_3d


def grouping_1st_step(image, sigma, kHard, nHard, lambdaHard2d, tauHard, NHard):
    height, width = image.shape
    all_groups = []
    

    # iterate through patches in the image with a step
    for x in range(0, height - kHard + 1, pHard):
        for y in range(0, width - kHard + 1, pHard):

            # grouping
            patch = image[x:x+kHard, y:y+kHard]
            search_window = get_search_window(image, x, y, patch_size=kHard, window_size=nHard)

            group_3d = build_3d_group(patch, search_window, sigma, lambdaHard2d, tauHard, NHard)
            N = len(group_3d)
            if N > 0:
                # check if its power of two
                if not (N & (N - 1)) == 0:
                    group_size = closest_power_of_two(N, NHard)
                    group_3d = group_3d[:group_size]
                all_groups.append((group_3d, (x, y)))

    return all_groups

#%% COLLABORATIVE FILTERING
#fast walsh-hadamard function
def apply_1d_transform(x):
    h = 1
    while h < len(x):
        for i in range(0, len(x), h * 2):
            for j in range(h):
                x[i + j], x[i + j + h] = x[i + j] + x[i + j + h], x[i + j] - x[i + j + h]
        h *= 2
    return x

def reverse_1d_transform(x): #reverse walsh hadamard
    n = len(x)
    x = apply_1d_transform(x)
    return x / n


def apply_2d_transform(patch, use_dct=False):
    if use_dct:
        return cv2.dct(patch)  # 2D DCT
    else:
        coeffs = pywt.wavedec2(patch, wavelet='bior1.5', level=2, mode='periodic')
        return coeffs

def reverse_2d_transform(patch, use_dct=False):
    if use_dct:
        return cv2.dct(patch, flags=cv2.DCT_INVERSE) #normalizacao ja foi feita
    else:
        coeffs = pywt.waverec2(patch, wavelet='bior1.5', level=2, mode='periodic')
        return coeffs

#%% 

im = skio.imread('./lena.tif') # original image
u = noise(im, sigma) # create noisy image

grouping_list = grouping_1st_step(u, sigma, kHard, nHard, lambdaHard2d, tauHard, NHard)

#%%

## collaborative filtering tests
group = grouping_list[0]

print('\nLast patch from original group:\n')
print(group[0][15])
plt.imshow(group[0][15],cmap='gray')
plt.title('Patch[15] from original group')
plt.show()

# t2d
transformed = np.array([apply_2d_transform(patch, use_dct=True) for patch in group[0]]) #t2d

#print(transformed[15])
plt.imshow(transformed[15],cmap='gray')
plt.title('Patch[15] after 2d DCT')
plt.show()

#t1d
for i in range(transformed.shape[-2]):
    for j in range(transformed.shape[-1]):
        transformed[:, i, j] = apply_1d_transform(transformed[:, i, j]) #t1d

#print(transformed[15])
plt.imshow(transformed[15], cmap='gray')
plt.title('Patch[15] after 2d DCT & 1D hadamard')
plt.show()


threshold = lambdaHard3d * sigma
after = hard_thresholding(transformed, threshold)
#after = transformed

#print(after[15])
plt.imshow(after[15], cmap='gray')
plt.title('Patch[15] after T3D & hard threshold')
plt.show()

# reverse 1d
for i in range(after.shape[-2]):
    for j in range(after.shape[-1]):
        after[:, i, j] = reverse_1d_transform(after[:, i, j]) #t1d

#print(after[15])
plt.imshow(after[15], cmap='gray')
plt.title('Patch[15] after reverse 1D hadamard')
plt.show()

#reverse t2d
final = np.array([reverse_2d_transform(patch, use_dct=True) for patch in after]) #t2d

print('\nLast patch from final group:\n')
print(final[15])
plt.imshow(final[15], cmap='gray')
plt.title('Patch[15] from final group (after reverse 3D)')
plt.show()

#%%
'''
here is python code that does this:
1) Compute 8x8 matrices that, when applied to a vector result in a 1d Bior1.5 
    trasnform
2) A function that applies bior1.5 and inverse bior1.5 to any dimension of a 
    tensor
3) A test section that demonstrates that.

To be clear the function get_Bior_matrices is called only once to 
prepare your program. The function apply_bior is called whenever needed. 
An example of 2D transform is also given.
'''
import numpy as np
import pywt

def get_Bior_matrices(N=8):
    directBior15_matrix=np.zeros((N,N))
    ss=N//2
    ls=[]

    while ss>0:
        ls=ls+[ss]
        ss=ss//2
    print (ls)   
    for k in range(N):
        inp=np.zeros(N)
        inp[k]=1
        tmp=inp
        out=[]
        for s in ls:
            #print (out,s)
            (a,b)=pywt.dwt(tmp,'bior1.5',mode='periodic')
            out=list(b[0:s])+out
            tmp=a[:s]
            #print ('sortie s=',s)
        out=list(a[:s])+out
        directBior15_matrix[k,:]=np.asarray(out)

    invBior15_matrix=np.linalg.inv(directBior15_matrix)
    return directBior15_matrix,invBior15_matrix

def apply_bior(V,M,dim):
    s=V.shape
    l=[0,1,2]
    l[dim]=0
    l[0]=dim
    smod=list(s)
    smod[dim]=s[0]
    smod[0]=s[dim]
    return (M@V.transpose(l).reshape(((M.shape[0]),-1))).reshape(smod).transpose(l)


#%% TEST
# The next line is done ONLY ONCE IN THE PROGRAM
B8,IB8=get_Bior_matrices(N=8)

v=np.random.randn(8,8,10)

vapp=apply_bior(v,B8,1) #Apply direct bior to dimension 1

print (vapp[6,:,7]-B8@v[6,:,7]) #check if the desired transform occured

#check for inverse

vappinv=apply_bior(vapp,IB8,1) # apply inverse bior to dimension 1
print (((vappinv-v)**2).sum())

# 2D bior transorf on dimensions 0 and 1. v is supposed size 8x8xN
v1d=apply_bior(v,B8,1)
v2d=apply_bior(v1d,B8,0) #see how
#%%
tmp=[0,0,1, 0,0,0,0,0]
ls=[4,2,1]
out=[]
for s in ls:
    #print (out,s)
    (a,b)=pywt.dwt(tmp,'bior1.5',mode='periodic')
    out=list(b[0:s])+out
    tmp=a[:s]