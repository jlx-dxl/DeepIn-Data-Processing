import cv2
import tifffile as tiff
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import xlwt


def tif_read(tif_dir):
    '''
    :param tif_dir: path & name of input TIF
    :return: tif
    '''
    tif = tiff.imread(tif_dir)
    print(tif.dtype)
    print(tif.shape)
    print(tif.max(), tif.min())
    return tif


def extract_img_from_tif(tif, img_index):
    '''
    :param tif: input TIF
    :param img_index: Sequence number of the picture to be extracted
    :return: img
    '''
    if tif.ndim == 3:
        img = tif[img_index, :, :]
    elif tif.ndim == 4:
        img = tif[img_index, :, :, :]
    else:
        print('tif ndim is wrong')
    return img


def change_tif_to_green(tif):
    '''
    :param tif: input TIF (ndim = 3)
    :return: green_tif
    '''
    green_tif = np.zeros([tif.shape[0], tif.shape[1], tif.shape[2], 3])
    for i in range(tif.shape[0]):
        green_tif[i, :, :, 1] = tif[i, :, :]
    return green_tif


def img_add_noise(img, Q, var):
    '''
    add noise and calculate value of snr,ssim,psnr
    :param img:
    :param Q:
    :param var:
    :return:
    '''
    # noise generation
    clean = img.astype(float)
    k = Q / 65534
    poisson = np.random.poisson(lam=clean * k, size=clean.shape).astype(float) / k
    gaussion = np.random.normal(0, var, clean.shape).astype(float)
    noisy_img = poisson + gaussion
    np.clip(noisy_img, 0, 65535).astype(np.uint16)
    # data analysis
    clean_im = img.astype(np.float32)  # float32改成别的数据格式算出来的值是有较明显区别的
    noisy_im = noisy_img.astype(np.float32)
    snrr = cal_snr(noisy_im, clean_im)
    noisy_img_ = cv2.normalize(noisy_im, None, 0.0, 1.0, cv2.NORM_MINMAX)
    clean_img_ = cv2.normalize(clean_im, None, 0.0, 1.0, cv2.NORM_MINMAX)
    np.clip(noisy_img_, 0.0, 1.0)
    np.clip(clean_img_, 0.0, 1.0)
    ssim_value = ssim(clean_img_, noisy_img_)
    psnr_value = psnr(clean_img_, noisy_img_)
    return noisy_img, snrr, ssim_value, psnr_value


def tif_add_noise(tif, Q, var, ws, n):
    '''
    :param tif:
    :param Q:
    :param var:
    :param ws:
    :param n:
    :return:
    '''
    result = np.zeros(tif.shape).astype('uint16')
    for i in tqdm(range(tif.shape[0]), desc='progressing', ncols=50):
        result[i, :, :], snrr, ssim_value, psnr_value = img_add_noise(tif[i, :, :], Q, var)
        ws.write((i + 2), n, snrr)
        ws.write((i + 2), n+6, ssim_value)
        ws.write((i + 2), n+12, psnr_value)
    return result


def cal_snr(noise_img, clean_img):
    '''
    :param noise_img: input noisy image
    :param clean_img: clean image used as reference
    :return: value of snr
    '''
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal ** 2
    clean_signal_2 = clean_signal ** 2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))
    return snrr


if __name__ == '__main__':

    list = np.array([1, 5, 10, 40, 150, 600])  # snr = -1, 0, 2, 4, 6, 8
    x = []
    y1 = []
    y2 = []
    y3 = []

    # read tif
    for i in range(4):
        n = 0
        i = i + 8
        wb = xlwt.Workbook()

        wsA = wb.add_sheet("A")
        wsA.write(0, 0, "snr")
        wsA.write(0, 6, "ssim")
        wsA.write(0, 12, "psnr")
        tif_dir = 'D:/#laboratory/Calcium Images/NAOMI/results/320Hz_' + str(i) + '/320Hz_' + str(i) + '_clean_A.tif'
        tif1 = tif_read(tif_dir)

        wsB = wb.add_sheet("B")
        wsB.write(0, 0, "snr")
        wsB.write(0, 6, "ssim")
        wsB.write(0, 12, "psnr")
        tif_dir = 'D:/#laboratory/Calcium Images/NAOMI/results/320Hz_' + str(i) + '/320Hz_' + str(i) + '_clean_B.tif'
        tif2 = tif_read(tif_dir)

        wsC = wb.add_sheet("C")
        wsC.write(0, 0, "snr")
        wsC.write(0, 6, "ssim")
        wsC.write(0, 12, "psnr")
        tif_dir = 'D:/#laboratory/Calcium Images/NAOMI/results/320Hz_' + str(i) + '/320Hz_' + str(i) + '_clean_C.tif'
        tif3 = tif_read(tif_dir)
        print('i:', i)

        for j in list:
            print('i:', i, ' Q:', j)
            # tif add noise
            wsA.write(1, n, str(j))
            wsA.write(1, n+6, str(j))
            wsA.write(1, n+12, str(j))
            noisy_tif = tif_add_noise(tif1, j, 1000, wsA, n)
            tiff.imsave(
                'D:/#laboratory/Calcium Images/NAOMI/results/320Hz_' + str(i) + '/320Hz_' + str(i) + '_noisy_A_Q' + str(
                    j) + '.tif', noisy_tif)

            wsB.write(1, n, str(j))
            wsB.write(1, n+6, str(j))
            wsB.write(1, n+12, str(j))
            noisy_tif = tif_add_noise(tif2, j, 1000, wsB, n)
            tiff.imsave(
                'D:/#laboratory/Calcium Images/NAOMI/results/320Hz_' + str(i) + '/320Hz_' + str(i) + '_noisy_B_Q' + str(
                    j) + '.tif', noisy_tif)

            wsC.write(1, n, str(j))
            wsC.write(1, n+6, str(j))
            wsC.write(1, n+12, str(j))
            noisy_tif = tif_add_noise(tif3, j, 1000, wsC, n)
            tiff.imsave(
                'D:/#laboratory/Calcium Images/NAOMI/results/320Hz_' + str(i) + '/320Hz_' + str(i) + '_noisy_C_Q' + str(
                    j) + '.tif', noisy_tif)
            n = n + 1
        wb.save('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_' + str(i) + '/320Hz_' + str(i) + '_snr_data.xls')

        # # extract img
        # clean = extract_img_from_tif(tif1, 2200)
        # # clean = cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX)
        # print(clean.dtype)
        # # cv2.imwrite('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/clean_2200_.png', clean)
        #
        # # add_noise_test & calculate snr
        # for k in tqdm(range(2000), desc='progressing', ncols=50):
        #     k = k + 1
        #     noisy_img = img_add_noise(clean, k, 1000)
        #     cv2.imwrite('D:/#laboratory/Calcium Images/NAOMI/results/320Hz_1/noisy_' + str(k) + '_.png', noisy_img)
        #     # calculate snr
        #     clean_im = cv2.normalize(clean, None, 0, 1, cv2.NORM_MINMAX)
        #     noisy_im = cv2.normalize(noisy_img, None, 0, 1, cv2.NORM_MINMAX)
        #     clean_im = clean_im.astype(np.float32)  # float32改成别的数据格式算出来的值是有较明显区别的
        #     noisy_im = noisy_im.astype(np.float32)
        #     snrr = cal_snr(noisy_im, clean_im)
        #     # psnr = psnr(clean_im, noisy_im)
        #     # ssim = ssim(clean_im, noisy_im)
        #     x.append(k)
        #     y1.append(snrr)
        #     # y2.append(psnr)
        #     # y3.append(ssim)
        # plt.plot(x, y1, c='r', label='snr')
        # # plt.plot(x, y2, c='b', label='psnr')
        # # plt.plot(x, y3, c='g', label='ssim')
        # plt.legend()
        # # plt.yticks(())
        # plt.title('snr')
        # plt.show()
