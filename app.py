from flask import Flask,render_template,url_for,request
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np


# import model and vectorizer
model=pickle.load(open('model.pkl','rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))

# stemmer 
stemmer=PorterStemmer()

# function that will predict
def predict(stri):
    # lower case
    st=stri.lower()
    # remove punctuation
    st=''.join([i for i in st.lower() if i not in string.punctuation])
    # remove stopwords
    st=[i for i in st.split() if i not in stopwords.words('english')]
    # stems
    st=' '.join([stemmer.stem(i) for i in st]) 
    #vectorize
    st=vectorizer.transform(pd.Series(st)).toarray()
    #predict
    pred=model.predict(st)
    if pred[0]==1:
        return 'https://www.worldbank.org/content/dam/photos/780x439/2020/nov-3/scam-alert-780x439-shutterstock_1012719211.gif'
    else:
        return 'https://media2.giphy.com/media/Y34V9u78XqfSWIHKJj/giphy.gif?cid=6c09b952336627414864e8696d791ef9132f76e9f45a70f0&rid=giphy.gif&ct=s'

#let' create flask app
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classifier' ,methods=['POST'])
def classifier():
    inp=str(request.form['text']).strip()
    # pass it into the function that we just made
    y=predict(inp)

    return render_template('index.html',output=y)

if __name__=='__main__':
    app.run(debug=True)