import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from train import Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 取得 IMDb 的單詞索引表

def predict_sentiment(review, model, word_index, max_length=200):
    # 1. 將 review 轉換為數字 ID
    words = review.lower().split()  # 轉換為小寫並分詞
    encoded_review = [word_index.get(word, 2) for word in words]  # 若單詞不存在，使用 <UNK> (2)
    
    # 2. 進行填充，確保長度一致
    encoded_review = pad_sequences([encoded_review], maxlen=max_length, padding='post', truncating='post')
    
    # 3. 使用模型進行預測
    prediction = model.predict(encoded_review)  # 取得預測機率
    
    # 確保 `prediction` 變數是 float，而不是 NumPy 陣列
    prediction = float(prediction[0][0])  
    # 4. 判斷情感結果
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    print(f"Predicted Sentiment: {sentiment} (Score: {prediction:.4f})")

    return sentiment
  
word_index = imdb.get_word_index()
model = load_model('model_with_attention.h5', custom_objects={'Attention': Attention})  # 載入訓練好的模型
# 測試範例影評
sample_review = "movie was disappointing and boring"
predict_sentiment(sample_review, model, word_index)
