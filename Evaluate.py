# 评价函数
# 评价指标1: 峰值信噪比(PSNR)
# 评价指标2: 结构相似度(SSIM)
# 注意：图像通道数可以任意，但是对比图像必须要保持相同

import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

img1 = cv2.imread('imgPred0.png')
img2 = cv2.imread('imgPred01.png')

MSE = mean_squared_error(img1, img2)
PSNR = peak_signal_noise_ratio(img1, img2)
SSIM = structural_similarity(img1, img2, multichannel=True)

print('MSE: ', MSE)
print('PSNR: ', PSNR)
print('SSIM: ', SSIM)