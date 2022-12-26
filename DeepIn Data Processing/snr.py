import os
import numpy as np
import math
import cv2


def cal_snr(noise_img,clean_img):
    noise_signal=noise_img-clean_img
    clean_signal=clean_img
    noise_signal_2=noise_signal**2
    clean_signal_2=clean_signal**2
    sum1=np.sum(clean_signal_2)
    sum2=np.sum(noise_signal_2)
    snrr=20*math.log10(math.sqrt(sum1)/math.sqrt(sum2))
    return snrr


clean = cv2.imread('F:/Laboratory/NAOMI/320Hz_try/clean_.png', 0)
for i in range(20):
    i = i + 1
    img = cv2.imread('F:/Laboratory/NAOMI/320Hz_try/noisy_' + str(i) + '.png', 0)
    snrr = cal_snr(img, clean)
    print(snrr,'dB')

