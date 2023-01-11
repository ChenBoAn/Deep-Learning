#TensorBoard: Tensorflow視覺化框架
#1.在訓練期間視覺化呈現監控指標
#2.視覺化模型架構
#3.視覺化啟動函數結果與梯度變化的直方圖
#4.以3D方式探索嵌入向量

from tensorflow import keras
from tensorflow.keras import layers, models, Input, callbacks
from tensorflow.keras.preprocessing import sequence

#處理IMDB資料
from keras.datasets import imdb

max_features = 2000 #做為特徵的常見單字數量
max_len = 500 #只看每篇評論前500個字

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

#建立模型
input = Input(shape=(max_len, ))
x = layers.Embedding(max_features, 128, input_length=max_len)(input)
x = layers.Conv1D(32, 7, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(32, 7, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(1)(x)

model = models.Model(input, x)

model.summary()

#編譯模型
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc'])

#使用TensorBoard回呼來訓練模型
callbacks_list = [callbacks.TensorBoard(log_dir="E:/DeepLearning/TensorBoard/record", #紀錄檔案的儲存位置
                                histogram_freq=1, #記錄每1個訓練週期啟動函數的結果直方圖
                                embeddings_freq=1)] #記錄每1個訓練週期嵌入向量資料

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks_list)

'''
cmd: tensorboard --logdir=E:/DeepLearning/TensorBoard/record
browser: http://localhost:6006
'''        

#呈現模型圖
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='E:/DeepLearning/TensorBoard/model.png') #show_shape=層圖形顯示shape資訊