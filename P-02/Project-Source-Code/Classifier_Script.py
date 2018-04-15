# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import nltk
import os
import sys
import numpy
import sklearn
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import scipy.stats as stats
import re
import string
import pandas
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn import metrics
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import sys
import json
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse



def removepuncs(convertstring):
	for punc in string.punctuation:
	      convertstring = convertstring.replace(punc,'')
	return convertstring

def preproccess(convertstring):
        ps = PorterStemmer()
	convertstring = re.sub('\s+', ' ', convertstring).strip()
	convertstring = re.sub("’", "", convertstring)
	convertstring = re.sub("“", "", convertstring)
	convertstring = re.sub(",", "", convertstring)
	convertstring = re.sub("\n", "", convertstring)
	convertstring=re.sub(' +'," ",convertstring)
	convertstring=re.sub('-',' ',convertstring)
	convertstring=re.sub(r"\s+", " ", convertstring)
	convertstring = removepuncs(convertstring)
	convertstring = re.sub(r'[0-9]', '', convertstring)
	convertstring = re.sub('\W+',' ', convertstring )
	convertstring = convertstring.lower()
	convertstring = nltk.word_tokenize(convertstring)
	convertstring = [word for word in convertstring if word not in stopwords.words('english')]
	convertstring = [[ps.stem(token) for token in sentence.split(" ")] for sentence in convertstring]
	convertstring = [item for sublist in convertstring for item in sublist]
	convertstring = [x.encode('utf-8') for x in convertstring]
	return convertstring

def readFile(filePath):
	dataset = open(filePath)
	return dataset.read()



def trainModel(train,train_labels,input2):
	if(input2 == 1):
		classifier_svm_lin = svm.SVC(kernel='linear')
		return classifier_svm_lin.fit(train, train_labels)

	if(input2 == 2):
		classifier_NB = MultinomialNB()
		return classifier_NB.fit(train, train_labels)

	if(input2 == 3):
		classifier_rf = RandomForestClassifier()
		return classifier_rf.fit(train, train_labels)

	if(input2 == 4):
		classifier_NB = GaussianNB()
		return classifier_NB.fit(train, train_labels)

	if(input2 == 5):
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
		return clf.fit(train, train_labels)

def getLabels(Trained_model, classifierName,test):
	if(classifierName=="GaussianNB"):
		predicts_labels = Trained_model.predict(test.toarray())
	else:
		predicts_labels = Trained_model.predict(test)
	return predicts_labels

def main():
    input = int(raw_input("Select The following 1.Unigram 2.bigram"))
    input1 = int(raw_input("Select The following 1.TfidfVectorizer 2.CountVectorizer"))
    input2 = int(raw_input("Select The Machine Learning algorithm 1.SVM 2.MultinomialNB 3.Random Forest 4.Gausian naive bayes 5.MLPClassifier"))
    Depressedvalues = readFile('/Users/rohitgs/Desktop/Depressed.txt')
    NotDepressedvalues = readFile('/Users/rohitgs/Desktop/Not-Depressed.txt')
    Depressedvalues = preproccess(Depressedvalues)
    bigram_list=[]
    bigram = True
    bigram_list_Depressed =[]
    bigram_list_NotDepressed =[]
    NotDepressedvalues = preproccess(NotDepressedvalues)
    Depressed = []
    NotDepressed = []
    Depressed.append('Depressed')
    NotDepressed.append('NotDepressed')
    if input == 1:
     Depressed =Depressed * len(Depressedvalues)
     NotDepressed = NotDepressed *len(NotDepressedvalues)
     df2 = pd.DataFrame({'Category':Depressed})
     df1 = pd.DataFrame({'Data':Depressedvalues})
     Depressed_dataframe = df1.join(df2 )
     print Depressed_dataframe
     df4 = pd.DataFrame({'Category':NotDepressed})
     df3 = pd.DataFrame({'Data':NotDepressedvalues})
     NonDepressed_Dataframe = df3.join(df4)
     print NonDepressed_Dataframe
     Total_Dataframe = Depressed_dataframe.append(NonDepressed_Dataframe)
     print Total_Dataframe
     #print Total_Dataframe
     if (input1 == 1):
      Vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=2, stop_words='english')
      X = Vectorizer.fit_transform(Total_Dataframe.pop('Data')).toarray()
     elif (input1 == 2):
      Vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2, stop_words='english')
      X = Vectorizer.fit_transform(Total_Dataframe.pop('Data')).toarray()
     print X
     r = Total_Dataframe[['Category']].copy()
     r = r.values.ravel()
     train, test,train_label,test_label  = train_test_split(X,r,test_size=0.2)
     print train[1]
     Trained_model = trainModel(train,train_label,input2)
     predicts_labels = Trained_model.predict(test)

     report = classification_report(test_label, predicts_labels)

     evalSentence = readFile('/Users/rohitgs/Desktop/input.txt')
     print "Given Input"
     print evalSentence
     print "Accuracy"
     print metrics.accuracy_score(test_label, predicts_labels)
     print(report)
     evalSentence = preproccess(evalSentence)
     df_eval = pd.DataFrame({'Data':evalSentence})
     print df_eval
     Y = Vectorizer.transform(df_eval.pop('Data')).toarray()
     #train1, test1  = train_test_split(X,test_size=0.0)
     predicts_labels = Trained_model.predict(Y)
     #predicts_labels= list(predicts_labels)
     DepressedCounter = 0
     NonDepressedCounter = 0
     for i in range(len(predicts_labels)):
       if predicts_labels[i] == 'Depressed':
        DepressedCounter = DepressedCounter + 1
       elif predicts_labels[i] =='NotDepressed':
        NonDepressedCounter = NonDepressedCounter +1
     #print DepressedCounter
     #print NonDepressedCounter
     if(DepressedCounter < NonDepressedCounter):
       print "The person is not depressed"
     elif(DepressedCounter > NonDepressedCounter):
       print "The person is depressed"
    if input ==2:
     for i in range(len(Depressedvalues)-1):
	      bigram_list_Depressed.append(Depressedvalues[i]+' '+Depressedvalues[i+1] )
     for i in range(len(NotDepressedvalues)-1):
	  	  bigram_list_NotDepressed.append(NotDepressedvalues[i]+' '+NotDepressedvalues[i+1])
     Depressed =Depressed * len(bigram_list_Depressed)
     NotDepressed = NotDepressed *len(bigram_list_NotDepressed)
     df2 = pd.DataFrame({'Category':Depressed})
     df1 = pd.DataFrame({'Data':bigram_list_Depressed})
     Depressed_dataframe = df1.join(df2 )
     #print Depressed_dataframe
     df4 = pd.DataFrame({'Category':NotDepressed})
     df3 = pd.DataFrame({'Data':bigram_list_NotDepressed})
     NonDepressed_Dataframe = df3.join(df4)
     #print NonDepressed_Dataframe
     Total_Dataframe = Depressed_dataframe.append(NonDepressed_Dataframe)
     print Total_Dataframe
     Total_Dataframe1 = Total_Dataframe['Data']
     print Total_Dataframe1
     if (input1 == 1):
      Vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words='english')
      X = Vectorizer.fit_transform(Total_Dataframe1)
     elif (input1 == 2):
      Vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, stop_words='english')
      X = Vectorizer.fit_transform(Total_Dataframe.pop('Data')).toarray()
     r = Total_Dataframe[['Category']].copy()
     r = r.values.ravel()
     train, test,train_label,test_label  = train_test_split(X,r,test_size=0.2)
     Trained_model = trainModel(train,train_label,input2)
     predicts_labels = Trained_model.predict(test)
     print metrics.accuracy_score(test_label, predicts_labels)
     report = classification_report(test_label, predicts_labels)
     print(report)
     evalSentence = readFile('/Users/rohitgs/Desktop/input.txt')
     print "Input String"
     print evalSentence
     evalSentence = preproccess(evalSentence)
     bigram_list = []
     for i in range(len(evalSentence)-1):
      bigram_list.append(evalSentence[i]+' '+evalSentence[i+1] )
     df_eval = pd.DataFrame({'Data':bigram_list})
     print df_eval
     Y = Vectorizer.transform(df_eval.pop('Data')).toarray()
     predicts_labels = Trained_model.predict(Y)
     predicts_labels= list(predicts_labels)
     #print predicts_labels
     DepressedCounter = 0
     NonDepressedCounter = 0
     for i in range(len(predicts_labels)):
       if predicts_labels[i] == 'Depressed':
        DepressedCounter = DepressedCounter + 1
       elif predicts_labels[i] =='NotDepressed':
        NonDepressedCounter = NonDepressedCounter +1
    # print DepressedCounter
     #print NonDepressedCounter
     if(DepressedCounter < NonDepressedCounter):
       print "The person is not depressed"
     elif(DepressedCounter > NonDepressedCounter):
       print "The person is depressed"

if __name__== "__main__":
  main()
