import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn import svm

from sklearn import datasets
file_path='smsspamcollection/SMSSpamCollection.txt'
data=pd.read_csv(file_path, delimiter='\t',header=None,skipinitialspace=True)

data=data.dropna(axis=0, how='any')

def pre_processing(sms):
    _str=re.sub("[^a-zA-Z]"," ",sms)
    words=_str.lower().split()
    new_str=[w for w in words if not w in set(stopwords.words("english"))]
    return (" ".join(new_str))

clean_data = []
for i in range(len(data[1])):
    clean_data.append(pre_processing(data[1][i]))
data[1] = clean_data

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(data.iloc[:,1], data.iloc[:,0],test_size=0.1, random_state=50)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
x_train = vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()

# SVM
clf=svm.SVC(kernel='linear',C=1.0)
clf.fit(x_train,y_train)

predicted=clf.predict(x_test)
accuracy=np.mean(predicted==y_test)
print ("SVM: " + str(accuracy))

# KNN
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

y_pred=neigh.predict(x_test)
from sklearn import metrics
score_test = metrics.accuracy_score(y_test, y_pred)
print("KNN: " + str(score_test))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
nb_pred = gnb.fit(x_train, y_train).predict(x_test)
accuracy = np.mean(nb_pred==y_test)
print("Naive Bayes: " + str(accuracy))