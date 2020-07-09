import random
import cv2
import glob

DATA_PATH = 'data'


def rotate(image):
    # 随机旋转角度
    angle = random.randrange(-45, 45)
    w = image.shape[1]
    h = image.shape[0]
    # 选择矩阵
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # 旋转
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC)
    return image


def augmention():
    for i in range(1, 6):
        class_folder = DATA_PATH + '/' + str(i)
        img_path_array = glob.glob(class_folder + "/*.jpg")
        for img_path in img_path_array:
            img = cv2.imread(img_path)
            for i in range(30):
                # 旋转
                img_rotate = rotate(img)
                # 水平翻转
                img_flip = cv2.flip(img_rotate, 1)
                rotate_save_path = img_path.split('.')[0] + '_rotate' + str(i) + '.' + img_path.split('.')[-1]
                flip_save_path = img_path.split('.')[0] + '_flip' + str(i) + '.' + img_path.split('.')[-1]
                cv2.imwrite(rotate_save_path, img_rotate)
                cv2.imwrite(flip_save_path, img_flip)


if __name__ == '__main__':
    augmention()