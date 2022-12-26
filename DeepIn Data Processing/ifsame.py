import cv2
from matplotlib import pyplot as plt

a = cv2.imread('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/clean_2200.png', 0).astype('uint16')
b = cv2.imread('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/clean_2200_.png', 0).astype('uint16')

c = a - b

# d = cv2.imread('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/noisy_1.png', 0)
#
# e = d - a

print(c)
print(c.max())
print(c.min())

plt.hist(c.ravel(), 256, [0, 256])
plt.show()
# plt.hist(e.ravel(),256,[0,256])
# plt.show()