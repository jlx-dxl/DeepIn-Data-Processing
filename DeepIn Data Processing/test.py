import tifffile as tiff
import numpy as np

test1 = tiff.imread('mu_1_00001.tif')
# test2 = tiff.imread('150Hz500_500pixels40mW_noisy_00002.tif')
print(test1.shape)
print(test1.max(),test1.min(),test1.dtype)
# print(test2.shape)
# print(test2.max(), test2.min(), test2.dtype)
# test = np.zeros([test1.shape[0] + test2.shape[0], test1.shape[1], test1.shape[2]])
# print(test.shape, test.dtype)
# for i in range(test1.shape[0]):
#     test[i,:,:] = test1[i,:,:]
#     print(i)
# for i in range(test2.shape[0]):
#     test[test1.shape[0]+i,:,:] = test2[i,:,:]
#     print(i)
# print(test.max(), test.min())
test2 = np.zeros([test1.shape[0],test1.shape[1],test1.shape[2],3])
# print(test2.shape)
for i in range(test1.shape[0]):
    test2[i,:,:,1] = test1[i,:,:]
    print(i)
tiff.imsave('mu_1_00001_green.tif', test2)

