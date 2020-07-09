from tensorflow.keras import layers, models, optimizers, backend
import os
import glob
import cv2
import numpy as np


# 不使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


DATA_PATH = 'data'
EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01


# 构建神经网络
def demo_inference(h, w, c):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, c)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='softmax'))
    return model


# 构建神经网络
def alex_inference(h, w, c):
    model = models.Sequential()
    # 第一层卷积网络，使用96个卷积核，大小为11x11步长为4
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), input_shape=(h, w, c), padding='same', activation='relu',
                            kernel_initializer='uniform'))
    # 池化层
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 使用池化层，步长为2
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第三层卷积，大小为3x3的卷积核使用384个
    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 第四层卷积,同第三层
    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 第五层卷积使用的卷积核为256个，其他同上
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    return model


# 训练网络
def training(input_data, label, epoch, batch_size, learning_rate):
    backend.clear_session()
    _, h, w, c = input_data.shape
    model = alex_inference(h, w, c)
    # model = demo_inference(h, w, c)
    optimizer = optimizers.Adadelta(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(input_data, label, batch_size=batch_size, epochs=epoch)
    model_folder = 'data/model'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # model_path = model_folder + '/demo_model.h5'
    model_path = model_folder + '/AlexNet_model.h5'
    model.save(model_path)


# 预测
def prediction(input_data):
    backend.clear_session()
    model_folder = 'data/model'
    model_path = model_folder + '/AlexNet_model.h5'
    # model_path = model_folder + '/demo_model.h5'
    model = models.load_model(model_path)
    print('Load model succeed!')
    predict = model.predict(input_data)
    # 找到概率最大的对应位置
    max_index = np.where(predict == np.max(predict))[1][0]
    return max_index


# 创建输入数据和标签
def create_input_label():
    input_data = []
    labels = []
    for i in range(1, 6):
        class_folder = DATA_PATH + '/' + str(i)
        img_path_array = glob.glob(class_folder + "/*.jpg")
        for img_path in img_path_array:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            img = cv2.resize(img, (227, 227))
            img_data = np.asarray(img, dtype=np.float32)
            # 数据归一化
            img_data = img_data / 127.5 - 1
            input_data.append(img_data)
            temp = np.zeros(5)
            temp[i - 1] = 1
            labels.append(temp)
    return input_data, labels


if __name__ == '__main__':
    input_data, labels = create_input_label()
    training(np.array(input_data), np.array(labels), EPOCH, BATCH_SIZE, LEARNING_RATE)
