import sys
from PyQt5.QtWidgets import QApplication, QWidget, \
        QMessageBox, QDesktopWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QCoreApplication
import cv2
import numpy as np
import preprocessing
import model
import random
import pygame
import time


GESTURE_LIST = ['布', '石头', '剪刀', '数字六', '数字三']


class myWindow(QWidget):
    def __init__(self, parent=None):
        super(myWindow, self).__init__(parent)
        # 记录当前处于的界面，0为主界面，1为游戏界面，2为胜负界面
        self.view_index = 0
        # 设置一个Timer，用于延迟显示摄像头数据
        self.timer_camera = QTimer()
        # 将Timer与实时显示摄像头内容的函数绑定
        self.timer_camera.timeout.connect(self.show_camera_data)
        # 获取摄像头
        self.cap = cv2.VideoCapture()
        # 设置相机大小
        self.cap.set(3, 640)
        self.cap.set(4, 360)
        # 初始化界面
        self.initUI()
        # 实例化倒数拍摄的线程对象
        self.downcount_thread = DownCountThread()
        # 将线程绑定到更新UI的函数中
        self.downcount_thread.my_signal.connect(self.get_hand_img)
        # 设置拍摄窗口大小
        self.width, self.height = 224, 224
        # 设置选取位置
        self.x0, self.y0 = 400, 60
        # 开启摄像头
        self.cap.open(0)
        # 获取摄像头数据
        self.get_camera_data()

        self.music_thread = MusicThread()
        self.music_thread.start()
        


    def initUI(self):
        self.create_realtime()
        self.create_prediction()
        self.create_gesture_hint()
        # 程序窗口大小固定
        self.setFixedSize(680, 960)

        self.center()
        self.setWindowIcon(QIcon('data/img/01.jpg'))
        self.setWindowTitle('Finger-guessing Game')
        palette1 = QPalette()
        palette1.setColor(palette1.Background, QColor(255, 255, 255))
        self.setPalette(palette1)

    # 创建捕获实时拍照的控件
    def create_realtime(self):
        self.label_countdown = QLabel('', self)
        self.label_countdown.setStyleSheet("QLabel{font-size:20px;font-weight: bold;}")
        self.label_countdown.resize(450, 30)
        self.label_countdown.move(20, 15)
        self.label_show_roi = QLabel(self)
        self.label_show_roi.setFixedSize(640, 360)
        self.label_show_roi.move(20, 50)
        background_img = QPixmap('data/img/相机.png')
        self.label_show_roi.setPixmap(background_img)
        self.label_show_roi.setAlignment(Qt.AlignCenter)

    # 创建预测结果的控件
    def create_prediction(self):
        self.label_prediction = QLabel('', self)
        self.label_prediction.setStyleSheet("QLabel{font-size:24px;font-weight: bold;color:red;}")
        self.label_prediction.setAlignment(Qt.AlignCenter)
        self.label_prediction.resize(320, 200)
        self.label_prediction.move(180, 430)
        self.label_prediction.setScaledContents(True)

    # 创建手势提示的控件
    def create_gesture_hint(self):
        # 提示
        label_hint = QLabel('请根据以下提示，在绿色框内做出相应手势', self)
        label_hint.setStyleSheet("QLabel{font-size:20px;font-weight: bold;color: red;}")
        label_hint.resize(450, 30)
        label_hint.move(20, 650)

        # 手势石头
        self.gesture_stone = QLabel(self)
        self.gesture_stone.setFixedSize(150, 150)
        self.gesture_stone.move(20, 700)
        background_img = QPixmap('data/img/石头.png')
        self.gesture_stone.setPixmap(background_img)
        self.gesture_stone.setScaledContents(True)
        self.gesture_stone.hide()
        self.label_stone = QLabel('石头', self)
        self.label_stone.setAlignment(Qt.AlignCenter)
        self.label_stone.setStyleSheet("QLabel{font-size:20px;font-weight: bold;}")
        self.label_stone.resize(50, 30)
        self.label_stone.move(70, 860)
        self.label_stone.hide()

        # 手势剪刀
        self.gesture_scissor = QLabel(self)
        self.gesture_scissor.setFixedSize(150, 150)
        self.gesture_scissor.move(265, 700)
        background_img = QPixmap('data/img/剪刀.png')
        self.gesture_scissor.setPixmap(background_img)
        self.gesture_scissor.setScaledContents(True)
        self.gesture_scissor.hide()
        self.label_scissor = QLabel('剪刀', self)
        self.label_scissor.setAlignment(Qt.AlignCenter)
        self.label_scissor.setStyleSheet("QLabel{font-size:20px;font-weight: bold;}")
        self.label_scissor.resize(50, 30)
        self.label_scissor.move(315, 860)
        self.label_scissor.hide()

        # 手势布
        self.gesture_paper = QLabel(self)
        self.gesture_paper.setFixedSize(150, 150)
        self.gesture_paper.move(510, 700)
        background_img = QPixmap('data/img/布.png')
        self.gesture_paper.setPixmap(background_img)
        self.gesture_paper.setScaledContents(True)
        self.gesture_paper.hide()
        self.label_paper = QLabel('布', self)
        self.label_paper.setAlignment(Qt.AlignCenter)
        self.label_paper.setStyleSheet("QLabel{font-size:20px;font-weight: bold;}")
        self.label_paper.resize(50, 30)
        self.label_paper.move(560, 860)
        self.label_paper.hide()

        # 手势3
        self.gesture_3 = QLabel(self)
        self.gesture_3.setFixedSize(150, 150)
        self.gesture_3.move(38.5, 700)
        background_img = QPixmap('data/img/03.png')
        self.gesture_3.setPixmap(background_img)
        self.gesture_3.setScaledContents(True)
        self.label_3 = QLabel('进入游戏', self)
        self.label_3.setAlignment(Qt.AlignCenter)
        self.label_3.setStyleSheet("QLabel{font-size:20px;font-weight: bold;}")
        self.label_3.resize(100, 30)
        self.label_3.move(58.5, 860)

        # 手势6
        self.gesture_6 = QLabel(self)
        self.gesture_6.setFixedSize(150, 150)
        self.gesture_6.move(510, 700)
        background_img = QPixmap('data/img/06.png')
        self.gesture_6.setPixmap(background_img)
        self.gesture_6.setScaledContents(True)
        self.label_6 = QLabel('退出程序', self)
        self.label_6.setAlignment(Qt.AlignCenter)
        self.label_6.setStyleSheet("QLabel{font-size:20px;font-weight: bold;}")
        self.label_6.resize(100, 30)
        self.label_6.move(535, 860)

    # 跳转到主页面
    def go_to_main(self):
        # 更新UI界面
        self.view_index = 0
        self.label_prediction.clear()
        self.label_3.setText('进入游戏')
        self.label_6.setText('退出程序')
        # 再次获取用户手势
        self.get_camera_data()

    # 跳转到游戏界面
    def go_to_play(self):
        # 更新UI界面
        self.view_index = 1
        self.gesture_3.hide()
        self.label_3.hide()
        self.gesture_6.hide()
        self.label_6.hide()
        self.gesture_stone.show()
        self.label_stone.show()
        self.gesture_paper.show()
        self.label_paper.show()
        self.gesture_scissor.show()
        self.label_scissor.show()
        self.label_prediction.clear()
        # 再次获取用户手势
        self.get_camera_data()

    # 跳转到结算界面
    def go_to_result(self, result):
        # 更新UI界面
        self.view_index = 2
        self.gesture_3.show()
        self.label_3.show()
        self.gesture_6.show()
        self.label_6.show()
        self.label_3.setText('再来一局')
        self.label_6.setText('返回主界面')
        self.gesture_stone.hide()
        self.label_stone.hide()
        self.gesture_paper.hide()
        self.label_paper.hide()
        self.gesture_scissor.hide()
        self.label_scissor.hide()
        self.label_prediction.clear()
        # 根据结果显示不同的图片
        if result == -1:
            background_img = QPixmap('data/img/lose.png')
            self.label_prediction.setPixmap(background_img)
        elif result == 0:
            background_img = QPixmap('data/img/平手.png')
            self.label_prediction.setPixmap(background_img)
        elif result == 1:
            background_img = QPixmap('data/img/win.png')
            self.label_prediction.setPixmap(background_img)
        # 再次获取用户手势
        self.get_camera_data()

    # 退出程序
    def go_to_exit(self):
        if self.cap.isOpened():
            self.cap.release()
        QCoreApplication.quit()

    # 界面跳转
    def next_step(self, index):
        # 处于主界面
        if self.view_index == 0:
            # 手势三
            if index == 4:
                # 跳转到游戏界面
                self.go_to_play()
            # 手势六
            elif index == 3:
                # 退出程序
                self.go_to_exit()
            # 其余手势
            else:
                # 重新打开摄像头，再次获取手势
                self.get_camera_data()
        # 处于游戏界面
        elif self.view_index == 1:
            # 随机生成软件出的手势
            app_gesture = random.randint(0, 2)
            # 用户出布
            if index == 0:
                # 软件出布
                if app_gesture == 0:
                    self.go_to_result(0)
                # 软件出石头
                elif app_gesture == 1:
                    self.go_to_result(1)
                # 软件出剪刀
                elif app_gesture == 2:
                    self.go_to_result(-1)
            # 用户出石头
            elif index == 1:
                # 软件出布
                if app_gesture == 0:
                    self.go_to_result(-1)
                # 软件出石头
                elif app_gesture == 1:
                    self.go_to_result(0)
                # 软件出剪刀
                elif app_gesture == 2:
                    self.go_to_result(1)
            # 用户出剪刀
            elif index == 2:
                # 软件出布
                if app_gesture == 0:
                    self.go_to_result(1)
                # 软件出石头
                elif app_gesture == 1:
                    self.go_to_result(-1)
                # 软件出剪刀
                elif app_gesture == 2:
                    self.go_to_result(0)
            # 用户出其余手势
            else:
                # 重新打开摄像头，再次获取手势
                self.get_camera_data()
        # 处于结算界面
        elif self.view_index == 2:
            # 手势三
            if index == 4:
                # 再次进入游戏界面
                self.go_to_play()
            # 手势六
            elif index == 3:
                # 进入到主界面
                self.go_to_main()
            # 其余手势
            else:
                # 重新打开摄像头，再次获取手势
                self.get_camera_data()

    # 开启摄像头获取数据
    def get_camera_data(self):
        self.timer_camera.start(30)
        # 重启开启倒数线程，进行拍照
        self.downcount_thread.count = 6
        self.downcount_thread.is_on = True
        self.downcount_thread.start()

    # 将摄像头内容显示到程序中
    def show_camera_data(self):
        flag, frame = self.cap.read()
        self.frame = cv2.flip(frame, 2)
        # 画出截取手势的框图
        cv2.rectangle(self.frame, (self.x0, self.y0), (self.x0 + self.width, self.y0 + self.height), (0, 255, 0))
        roi = self.frame
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # 显示到程序中
        show_roi = QImage(roi.data, roi.shape[1], roi.shape[0], QImage.Format_RGB888)
        self.label_show_roi.setPixmap(QPixmap.fromImage(show_roi))

    # 获取手势图片
    def get_hand_img(self, count):
        if int(count) > 0:
            self.label_countdown.setText('将在' + count + '秒后获取您的手势')
        elif int(count) == 0:
            # self.label_prediction.setText('您的手势预测结果为：预测中......\n')
            # 获取手势框图
            ret = self.frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width]
            # 获取训练数据时使用
            # cv2.imwrite('1.jpg', ret)
            # 进行肤色检测
            ret = preprocessing.get_hand(ret)
            # 转换到YCbCr空间后预测
            ret = cv2.cvtColor(ret, cv2.cv2.COLOR_BGR2YCR_CB)
            result_string, index = self.prediction(ret)
            print('预测结果: ' + result_string)
            self.next_step(index)

    # 调用模型预测手势
    def prediction(self, img):
        img = cv2.resize(img, (227, 227))
        img_data = np.asarray(img, dtype=np.float32)
        img_data = img_data / 127.5 - 1
        h, w, c = img_data.shape
        img_data = img_data.reshape((1, h, w, c))
        index = model.prediction(img_data)
        return GESTURE_LIST[index], index

    # 关闭按钮事件
    def closeEvent(self, QCloseEvent):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

    # 程序窗口居中显示
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

# 音乐线程
class MusicThread(QThread):  

    def __init__(self):
        super(MusicThread, self).__init__()
        self.is_on = True

    # 线程执行函数
    def run(self):
        filepath = 'data/bgm.mp3'
        pygame.mixer.init()
        # 加载音乐
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play(start=0.0)
        time.sleep(300)
        


# 倒数线程，用于倒数拍照
class DownCountThread(QThread):
    # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    my_signal = pyqtSignal(str)

    def __init__(self):
        super(DownCountThread, self).__init__()
        self.count = 6
        self.is_on = True

    # 线程执行函数
    def run(self):
        while self.is_on:
            self.count -= 1
            # 释放自定义的信号
            self.my_signal.emit(str(self.count))
            # 本线程睡眠1秒
            self.sleep(1)
            if self.count == -1:
                self.is_on = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = myWindow()
    win.show()
    sys.exit(app.exec())
