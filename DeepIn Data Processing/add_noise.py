import cv2
import numpy as np
from skimage.util import random_noise
import scipy.stats as stats
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def cal_snr(noise_img, clean_img):
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal ** 2
    clean_signal_2 = clean_signal ** 2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))
    return snrr


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


clean_img = cv2.imread('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/clean_2200.png', 0).astype('uint16')
print(clean_img.dtype)

for i in range(1000):
    i = i + 1
    noise_img = add_noise(clean_img, i, 1000)
    print(i)
    # clean = cv2.imread('F:/Laboratory/NAOMI/320Hz_try/noisy_' + str(i) + '.png', 0).astype('uint16')
    # clean = cv2.normalize(clean, None, 0, 65535, cv2.NORM_MINMAX)

    # print(noise_img.dtype)
    # print(noise_img.shape)
    # # print(clean.ndim)
    # print(noise_img.max())
    # print(noise_img.min())
    # cv2.imshow('clean', clean)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # noise = np.zeros(clean.shape)
    #

    #
    # noise_img = add_noise(clean, 10) + np.random.normal(0, 50 ** 0.5, clean.shape)
    # noise_img = cv2.normalize(noise_img, None, 0, 255, cv2.NORM_MINMAX)
    # print(noise.dtype)
    # print(noise.shape)
    # # print(noise.ndim)
    # print(noise.max())
    # print(noise.min())

    # noise_img = clean + np.random.normal(0, 50 ** 0.5, clean.shape) + np.random.poisson(lam=50, size=clean.shape)
    # noise_img = cv2.normalize(noise_img, None, 0, 255, cv2.NORM_MINMAX)
    # print(noise_img.dtype)
    # print(noise_img.shape)
    # print(noise_img.ndim)
    # print(noise_img.max())
    # print(noise_img.min())
    #
    # cv2.imwrite('F:/Laboratory/NAOMI/320Hz_try/noisy_4.png', noise_img)

    # gaussion = np.random.normal(0, 50 ** 0.5, clean.shape)
    # gaussion = cv2.normalize(gaussion, None, 0, 1, cv2.NORM_MINMAX).astype('uint16')
    # print(gaussion.max())
    # print(gaussion.min())
    # poisson = random_noise(clean, mode='poisson').astype('uint16')
    # print(poisson.max())
    # print(poisson.min())
    # cv2.imshow('noisy',noise_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # noise_img = clean + np.random.normal(0, 50 ** 0.5, clean.shape) + np.random.poisson(lam=50, size=clean.shape)
    # noise_img = cv2.normalize(noise_img, None, 0, 255, cv2.NORM_MINMAX)

    # cv2.imwrite('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/noisy_' + str(i) + '.png', noise_img)
    noise_img = cv2.normalize(noise_img, None, 0, 1, cv2.NORM_MINMAX)
    clean_img_ = cv2.normalize(clean_img, None, 0, 1, cv2.NORM_MINMAX)
    snrr = cal_snr(noise_img, clean_img_)
    print('snr:', snrr,'dB')
    print('ssim:', ssim(noise_img, clean_img_))
    print('psnr:', psnr(noise_img, clean_img_), '\n')
