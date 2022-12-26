import tifffile as tiff
import numpy as np
import cv2


def tif_read(name):
    test = tiff.imread(name + '.tif')
    print(test.dtype)
    print(test.shape)
    print(test.max(), test.min())
    return test


def alter_green(tif):
    test = np.zeros([tif.shape[0], tif.shape[1], tif.shape[2], 3])
    for i in range(tif.shape[0]):
        test[i, :, :, 1] = tif[i, :, :]
    return test


# for i in range(200):
#     print(i)
#     name = 'mu_' + str(i + 1) + '_00001'
#     tif = tif_read(name)
#     # tif_green = alter_green(tif)
#     sample = tif[159,:,:]
#     sample = (sample / sample.max()) * 255
#     cv2.imwrite(name + '_255_160.png', sample)

# clean_tif = tif_read('F:/Laboratory/NAOMI/320Hz_500pixels_1/100_00001')
# # clean_tif_green = alter_green(clean_tif)
# sample = clean_tif[1400,:,:]
# sample = 9 * cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX)
# cv2.imwrite('F:/Laboratory/NAOMI/320Hz_500pixels_1/100_00001_1400.png', sample)

clean_tif = tif_read('F:/Laboratory/NAOMI/320Hz_try/320Hz_1_noisy_1_00001')
# clean_tif_green = alter_green(clean_tif)
sample = clean_tif[1599,:,:]
sample = 10 * cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite('F:/Laboratory/NAOMI/320Hz_try/noisy_1.png', sample)
