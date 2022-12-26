import cv2
import tifffile as tiff
from PIL import Image

test = tiff.imread('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/320Hz_1noisy_A_Q_1.tif')
print(test.dtype)
print(test.shape)
print(test.ndim)



# for i in range(test.shape[0]):
#     sample = test[i,:,:]
#     # dist = cv2.normalize(sample, None, 1, 255, cv2.NORM_MINMAX)
#     # print(dist.max())
#     # print(dist.min())
#     # dist = (sample / sample.max()) * 255
#     cv2.imwrite('mu_1_00001_'+str(i+1)+'.jpg', sample)
dist = test[2200,:,:]
# dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/Q_1_.png', dist)
# sample = test[299,:,:,:]
# dist = (sample / sample.max()) * 255
# cv2.imwrite('sample1300.png', dist)
# sample = test[399,:,:,:]
# dist = (sample / sample.max()) * 255
# cv2.imwrite('sample1400.png', dist)
# sample = test[499,:,:,:]
# dist = (sample / sample.max()) * 255
# cv2.imwrite('sample1500.png', dist)