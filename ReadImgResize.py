# 读取图片
# 利用三次样条插值缩小到指定分辨率


import os
import cv2


readPath = "../Datasets/Mini-ImageNet/images/"
savePath = "../Datasets/Mini-ImageNet/imagesDx4/"

img = cv2.imread(readPath + "n0153282900000005.jpg", cv2.IMREAD_ANYCOLOR)
# 采用 bicubic 双三次插值
InterCubic = cv2.resize(img, (2500,2500), interpolation=cv2.INTER_CUBIC)
# 采用最近邻插值
InterNearest = cv2.resize(img, (2500,2500), interpolation=cv2.INTER_NEAREST)

cv2.imwrite(savePath + "InterCubic.jpg", InterCubic)
cv2.imwrite(savePath + "InterNearest.jpg", InterNearest)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

