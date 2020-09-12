# 评价函数
# 评价指标1: 峰值信噪比(PSNR)
# 评价指标2: 结构相似度(SSIM)
# 注意：图像通道数可以任意，但是对比图像必须要保持相同

import cv2
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

img1 = cv2.imread('imgPred0.png')
img2 = cv2.imread('imgPred01.png')

MSE = compare_mse(img1, img2)
PSNR = compare_psnr(img1, img2)
SSIM = compare_ssim(img1, img2, multichannel=True)

print('MSE: ', MSE)
print('PSNR: ', PSNR)
print('SSIM: ', SSIM)