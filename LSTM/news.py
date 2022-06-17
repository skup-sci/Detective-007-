from flask import Flask, escape, request, render_template
import numpy as np
import pandas as pd
import nltk
import re
from os import sep
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Embedding, LSTM, Dropout, GlobalAveragePooling1D, Flatten, Dense

#-----------------------------------------Model Training and Prediction----------------------------------------------------------------
inputdir1 ="Data"
fname1 = "fake_news_basic.csv"
filename1 = inputdir1 + sep + fname1
fake_news=pd.read_csv(filename1)
inputdir2 = "Data"
fname2 = "true_news_basic.csv"
filename2 = inputdir2 + sep + fname2
true_news=pd.read_csv(filename2)
# make both length equal
if len(fake_news) > len(true_news):
    fake_news = fake_news.head(len(true_news))
elif len(true_news) > len(fake_news):
    true_news = true_news.head(len(fake_news))
true_news['label']=1
fake_news['label']=0
news = pd.concat([fake_news, true_news])
x= news.drop('label',axis=1)
y = news['label']
temp = x.copy()
temp.reset_index(inplace=True)
# stopwords
stopwords_hindi =[
    'अत','अपना','अपनी','अपने','अभी','अंदर','आदि','आप','इत्यादि','इन','इनका','इन्हीं','इन्हें','इन्हों','इस','इसका','इसकी','इसके','इसमें',
    'इसी','इसे','उन','उनका','उनकी','उनके','उनको','उन्हीं','उन्हें','उन्हों','उस','उसके','उसी','उसे','एक','एवं','एस','ऐसे','और','कई',
    'कर','करता','करते','करना','करने','करें','कहते','कहा','का','काफ़ी','कि','कितना','किन्हें','किन्हों','किया','किर','किस','किसी','किसे','की',
    'कुछ','कुल','के','को','कोई','कौन','कौन','बही','बहुत','बाद','बाला','बिलकुल','भी','भीतर','मगर','मानो','मे','में','यदि','यह','यहाँ','यही',
    'या','यिह','ये','रखें','रहा','रहे','ऱ्वासा','लिए','लिये','लेकिन','व','वग़ैरह','वर्ग','वह','वहाँ','वहीं','वाले','वुह','वे','सकता','सकते','सबसे',
    'सभी','साथ','साबुत','साभ','सारा','से','सो','संग','ही','हुआ','हुई','हुए','है','हैं','हो','होता','होती','होते','होना','होने']
stopwords= stopwords_hindi
input_array=np.array(temp['short_description'])
corpus = []
for i in range(0,len(input_array)):
    review = input_array[i]
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords_hindi)]
    review = ' '.join(review)
    corpus.append(review)
voc_size=5000
from tensorflow.keras.preprocessing.text import one_hot
one_hot_repr =[one_hot(words,voc_size) for words in corpus]
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(one_hot_repr,padding='post',maxlen =20)
embed_dim = 40
model = Sequential([
    Embedding(voc_size,embed_dim,input_length=20),
    Bidirectional(LSTM(100)),
    #Flatten(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2), 
    Dense(32, activation='relu'),
    Dense(1,activation='sigmoid')
    ])
model.compile(loss='binary_crossentropy',optimizer ='adam',metrics =['accuracy'])
x = np.array(padded)
y = np.array(y)
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y,test_size=0.3,random_state=0)
history = model.fit(trainX,trainY, epochs =10, validation_data=(testX,testY),batch_size=64)
#print(testX[0])
pred = model.predict(testX)
binary_predictions = []
for i in pred:
    if i >= 0.5:
        binary_predictions.append(1)
    else:
        binary_predictions.append(0) 
print(binary_predictions)
#print(testY)
#------------------------------------------Flask start----------------------------------------------------------
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        new_array=[news]
        corpusnew = []
        for i in range(0,len(new_array)):
            review = new_array[i]
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords_hindi)]
            review = ' '.join(review)
            corpusnew.append(review)
        one_hot_reprnew =[one_hot(words,voc_size) for words in corpusnew]
        paddednew = pad_sequences(one_hot_reprnew,padding='post',maxlen =20)
        xnew = np.array(paddednew)
        #print(xnew)
        pred=model.predict(xnew)
        if pred>=0.5:
            binary_predictions='True'
        else:
            binary_predictions='False'
        predict=binary_predictions

        return render_template("prediction.html", prediction_text="News headline is -> {}".format(predict))
    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run()




