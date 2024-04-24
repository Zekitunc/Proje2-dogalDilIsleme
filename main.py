# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 00:23:28 2024

@author: ZEKİ&EMRE
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk


nltk.download('punkt')
nltk.download('stopwords')

dataset = pd.read_csv(r'C:\Users\Cansuk\Masaüstü\Proje2\YapayZEKA_Train.csv', delimiter=',');


# Duygu etiketlerini X ve y olarak ayırma
X = dataset['DİZE']
y = dataset['DUYGU']

# Eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vektörlerini oluşturma
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('turkish'))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Modeli oluşturma ve eğitme
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Modeli test etme
y_pred = svm_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)



# Kullanıcı girişi
user_input = input("Lütfen duygusunu analiz etmek istediğiniz metni giriniz:")

# TF-IDF vektörlerini oluşturma
input_text_vector = tfidf_vectorizer.transform([user_input])

# Modeli kullanarak duygu tahmini yapma
predicted_sentiment = svm_model.predict(input_text_vector)[0]

# Tahmin sonucunu ekrana yazdırma
print("Girilen metnin duygusu:", predicted_sentiment)