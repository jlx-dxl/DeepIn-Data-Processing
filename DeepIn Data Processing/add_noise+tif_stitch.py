import cv2
import numpy as np
import tifffile as tiff
from skimage.util import random_noise
# import scipy.stats as stats
# import math
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr

dir = '320Hz_1'
tif1 = tiff.imread('./' + dir + '/' + dir + '_clean_00001.tif')
tif2 = tiff.imread('./' + dir + '/' + dir + '_clean_00002.tif')
print(tif1.dtype)
print(tif1.shape)
print(tif1.max(), tif1.min())
print(tif2.dtype)
print(tif2.shape)
print(tif2.max(), tif2.min())
# tif1 = cv2.normalize(tif1, None, 0, 65535, cv2.NORM_MINMAX)
# tif2 = cv2.normalize(tif2, None, 0, 65535, cv2.NORM_MINMAX)
# tiff.imsave('./' + dir + '/' + dir + '_clean_A_.tif', tif1)
# tiff.imsave('./' + dir + '/' + dir + '_clean_B_.tif', tif2)
# print(tif1.max(), tif1.min())
# print(tif2.max(), tif2.min())


def add_noise(img, Q, var):
    # clean = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX).astype('uint16')
    clean = img
    k = Q / 65534
    poisson = (random_noise(clean * k, mode='poisson') / k)
    gaussion = np.random.normal(0, var, clean.shape)
    gaussion = cv2.normalize(gaussion, None, 0, 1, cv2.NORM_MINMAX)
    noise = cv2.normalize((gaussion + poisson), None, 0, 65535, cv2.NORM_MINMAX)
    noisy_img = cv2.normalize((noise + clean), None, 0, 65535, cv2.NORM_MINMAX).astype('uint16')
    return noisy_img

list = np.array([1,5,10,50,100])
tif1_result = np.zeros(tif1.shape).astype('uint16')
tif2_result = np.zeros(tif1.shape).astype('uint16')

# for j in list:
#     for i in range(tif1.shape[0]):
#         tif1_result[i, :, :] = add_noise(tif1[i, :, :], j, 1000)
#         tif2_result[i, :, :] = add_noise(tif2[i, :, :], j, 1000)
#     tiff.imsave('./' + dir + '/' + dir + 'noisy_A_Q' + str(j) + '.tif', tif1_result)
#     tiff.imsave('./' + dir + '/' + dir + 'noisy_B_Q' + str(j) + '.tif', tif2_result)
#     print(j)

for i in range(tif1.shape[0]):
    tif1_result[i, :, :] = add_noise(tif1[i, :, :], 1, 1000)
    tif2_result[i, :, :] = add_noise(tif2[i, :, :], 1, 1000)
tiff.imsave('./' + dir + '/' + dir + 'noisy_A_Q_1.tif', tif1_result)
tiff.imsave('./' + dir + '/' + dir + 'noisy_B_Q_1.tif', tif2_result)