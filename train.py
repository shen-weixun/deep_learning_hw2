import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
# 設定詞彙表大小
vocab_size = 20000  # 選取最常用的 20,000 個單詞
max_length = 200    # 每條評論統一長度


# 加載 IMDb 數據集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 進行填充，使所有評論長度一致
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# 劃分訓練集與驗證集 (80% 訓練, 20% 驗證)
x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 自定義 Attention 層
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  # 計算關注分數
        a = K.softmax(e, axis=1)  # 計算權重
        return x * a  # 加權輸出

# 構建帶注意力機制的 LSTM
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(32, return_sequences=True),  # 讓 LSTM 輸出所有時間步的結果
    Attention(),  # 加入注意力機制
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()
# 編譯與訓練
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_new, y_train_new, epochs=5, batch_size=64, validation_data=(x_val, y_val))

# 評估
loss, acc = model.evaluate(x_test, y_test)
print(f"測試集準確率 (帶注意力): {acc:.4f}")

# 取得模型在測試集上的預測結果 (機率值)
y_pred_probs = model.predict(x_test)

# 確保維度正確 (取最後一個時間步的輸出)
if len(y_pred_probs.shape) == 3:  
    y_pred_probs = y_pred_probs[:, -1, 0]  # 取最後一個時間步的結果

# 轉換為 0 或 1
y_pred = (y_pred_probs > 0.5).astype(int)

# 確保 `y_test` 維度正確
y_test = np.array(y_test).reshape(-1)  # 確保是 1D 陣列
y_pred = y_pred.reshape(-1)  # 轉換為 1D 陣列



# 計算各種指標
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 計算 Precision-Recall 曲線
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_probs)
# 計算平均精度 (mAP)
mAP = auc(recalls, precisions)

print(f"平均精度 (mAP): {mAP:.4f}")
print(f"精確率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分數 (F1-score): {f1:.4f}")

model.save('model_with_attention.h5')