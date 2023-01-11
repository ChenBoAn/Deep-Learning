#實作RNN前向傳遞範例:
#將序列化向量作為輸入，然後編碼成2D張量，shape=(timesteps 時間點, input_features 輸入特徵)
#RNN會在各時間點上進行循環，到達每個時間點時，考慮該時間點t的當前狀態與輸入(shape=(input_features,))，並將它們一起處理以取得該時間點的輸出，
#然後將此輸出定為新狀態，以供下一時間點使用。
#對於第一個時間點，由於前一個輸出未定義，因此無當前狀態，我們將狀態初始化為"全零向量"，稱為神經網路初始狀態(initial state)

'''
RNN pseudocode:

state_t = 0                                #t點的狀態
for input_t in input_sequence:             #迭代處理序列化向量中的各元素
    output_t = rnn(input_t, state_t)       #當前狀態與輸入送入RNN取得輸出
    state_t = output_t                     #將當前輸出設定成下一次迭代的狀態

#函數rnn()，就是將輸入與狀態轉換為輸出，作法是參數化兩個權重矩陣W和U以及偏移向量b的處理，類似前饋式神經網路(feedforward networks)中密集連接層的轉換。

state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
'''

#以Numpy實作簡單的RNN
import numpy as np

timesteps = 100 #輸入序列資料中的時間點數量
input_features = 32 #輸入特徵空間的維度
output_features = 64 #輸出特徵空間的維度

inputs = np.random.random((timesteps, input_features)) #輸入資料: 隨機產生數值

state_t = np.zeros((output_features, )) #神經網路初始狀態: 全零向量

#建立隨機權重矩陣
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
#建立隨機偏移向量
b = np.random.random((output_features, ))

successive_outputs = [] #用來儲存各時間點的輸出
for input_t in inputs: #input_t是一個向量，shape=(input_features,)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) #結合輸入與當前狀態，取得當前輸出張量(雙曲正切值)
    print(output_t.shape) #某時間點的輸出張量shape=(特徵數,)
    successive_outputs.append(output_t) #將當前輸出存入list
    state_t = output_t #以當前輸出做為下一時間點之神經網路狀態

final_output_sequence = np.array(successive_outputs) #串接所有時間點之輸出做為最終輸出(2D張量)，shape=(timesteps 時間點數量, output_features 輸出特徵)
print(final_output_sequence.shape) #最後的輸出張量

#RNN的特徵在於它的階躍函數:
#output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) 

#輸出張量中每個時間點t包含關於輸入序列資料中的時間點0 ~ t的資訊，也就是與整個過去有關。
#因此在多數情形下，不需要輸出的完整序列化資料，只需最後一個輸出(循環結束時的output_t)，因為它已包含有關整個序列化資料的資訊。