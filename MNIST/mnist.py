from pickletools import optimize
from keras.datasets import mnist #從keras的dataset匯入mnist資料集

#用mnist.load_data()取得mnist資料集，並存成tuple
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#(train_images, train_labels), (test_images, test_labels) = ((train_images, train_labels), (test_images, test_labels))
#只要有逗號，沒有小括號也視作tuple

#train data
print(train_images.shape)
print(len(train_labels))
print(train_labels)

#test data
print(test_images.shape)
print(len(test_labels))
print(test_labels)

'''
神經網路架構
'''
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    #密集層(Dense Layers) = 全連接層(fully connected)
    #前後層神經元全部皆彼此連接在一起
    layers.Dense(512, activation='relu'), 
                #該層寬度(神經單元個數)
    layers.Dense(10, activation='softmax') 
    #一個有10個輸出的softmax層，輸出一個含有10個機率評分(probability scores)的陣列(機率總和為1)
    #每個評分為目前數字圖片可能屬於哪一個數字類別的機率
])

'''
編譯步驟
'''
model.compile(optimizer="rmsprop", #指定優化器: 神經網路根據其輸入資料及損失函數值而自行更新的機制
        loss="sparse_categorical_crossentropy", #指定損失函數: 衡量神經網路在訓練資料上的表現，並引導網路朝正確的方向修正
        metrics=["accuracy"]) #指定評量準則: 圖片是否分類至正確類別

'''
準備圖片資料
'''
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

'''
訓練神經網路模型
'''
model.fit(train_images, train_labels, epochs=5, batch_size=128)

'''
用模型進行預測
'''
test_digits = test_images[0:10]
predictions = model.predict(test_digits)

print(predictions[0]) #預測0~9個別機率
print(predictions[0].argmax()) #預測結果
print(predictions[0][7]) #預測是數字7的機率
print(test_labels[0]) #實際結果

'''
評估模型在測試集上的表現
'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")