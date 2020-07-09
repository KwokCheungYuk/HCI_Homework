import cv2
import numpy as np


# 腐蚀膨胀操作
def erosion_dilation(img):
    # 设置卷积核
    kernel = np.ones((3, 3), np.uint8)
    # 腐蚀
    erosion_img = cv2.erode(img, kernel)
    # 膨胀
    dilation_img = cv2.dilate(erosion_img, kernel)
    return dilation_img


# 计算轮廓面积
def cal_area(contour):
    area = cv2.contourArea(contour)
    return area


# 轮廓检测
def contours_detect(img):
    # canny检测
    canny_img = cv2.Canny(img, 50, 200)
    # 寻找轮廓
    h = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = h[0]
    # 将轮廓从大到小排序
    contours.sort(key=cal_area, reverse=True)
    # 创建黑色背景
    result = np.ones(img.shape, np.uint8)
    # 选取最大轮廓绘制白色边缘
    cv2.drawContours(result, contours[0], -1, (255, 255, 255), 1)
    return result


# 手部提取
def get_hand(img):
    # 转换为YCbCr空间后，提取Cr分量
    YCbCr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCbCr_img)
    cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)
    # 肤色检测，Otsu二值化得到手部的mask
    _, hand_mask = cv2.threshold(cr_blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 腐蚀膨胀
    e_d_hand_mask = erosion_dilation(hand_mask)
    # 提取手部
    hand_img = cv2.bitwise_and(img, img, mask=e_d_hand_mask)
    # 轮廓检测
    edge_img = contours_detect(hand_img)
    return hand_img


if __name__ == '__main__':
    path = input()
    img = cv2.imread(path)
    hand_img = get_hand(img)
