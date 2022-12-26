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


def img_add_noise(img, Q, var):
    '''
    :param img: input img
    :param Q: number of photon
    :param var: gaussion noise variance
    :return: noisy_img
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


def tif_add_noise(tif, Q, var, ws, n, j):
    '''
    :param tif:
    :param Q:
    :param var:
    :param ws:
    :param n:
    :return:
    '''
    result = np.zeros(tif.shape).astype('uint16')
    for m in tqdm(range(tif.shape[0]), desc='progressing', ncols=50):
        result[m, :, :], snrr, ssim_value, psnr_value = img_add_noise(tif[m, :, :], Q, var)
        ws[0].write((m + 1 + j * 3201), n, snrr)
        ws[1].write((m + 1 + j * 3201), n, ssim_value)
        ws[2].write((m + 1 + j * 3201), n, psnr_value)
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
    tif = []
    ws = []
    x = []
    y1 = []
    y2 = []
    y3 = []

    # take 320Hz-10 as sample
    wb = xlwt.Workbook()
    wsA = wb.add_sheet("snr")
    ws.append(wsA)
    wsB = wb.add_sheet("ssim")
    ws.append(wsB)
    wsC = wb.add_sheet("psnr")
    ws.append(wsC)
    tif_dir = 'F:/Laboratory/NAOMI/320Hz_10/320Hz_10_clean_A.tif'
    tif.append(tif_read(tif_dir))
    tif_dir = 'F:/Laboratory/NAOMI/320Hz_10/320Hz_10_clean_B.tif'
    tif.append(tif_read(tif_dir))
    tif_dir = 'F:/Laboratory/NAOMI/320Hz_10/320Hz_10_clean_C.tif'
    tif.append(tif_read(tif_dir))

    for j in range(3):
        for i in range(200):
            print(j + 1, i + 1)
            if i < 100:
                Q = 1 + i
            else:
                Q = 100 + 9 * (i - 99)
            ws[j].write(0, i, Q)
            noisy_tif = tif_add_noise(tif[j], Q, 1000, ws, i, j)

    wb.save('F:/Laboratory/NAOMI/snr_Q.xls')
