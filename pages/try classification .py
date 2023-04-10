import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from models.classification import classifier
from models.classification import predict
im = Image.open('bot.png')
st.set_page_config(
    page_title="nocodemodelbuilder",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
my_bar = st.progress(0)
for percent_complete in range(80):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)



if 'file' in st.session_state:
    st.write(st.session_state['file'])

if 'n_estimators' not in st.session_state:
    st.session_state['n_estimators'] = ""


n_estimators_o = st.number_input('no of estimators: ',min_value=5)
if n_estimators_o:
    st.session_state['n_estimators'] = int(n_estimators_o)
    st.write(st.session_state['n_estimators'])

df = st.session_state['file']




model = classifier(df,n_estimators_o)

st.write(model['lc_acc'])
st.write(model['random_acc'])
st.write(model['naive_acc'])
st.write(model['svm_acc'])

X_coloumns  = list(df.iloc[:, :-1].columns.values)
st.write(X_coloumns)
a = len(X_coloumns) 
list_of_variables = []
for i in range(a):
    var = i
    val = st.number_input('Enter your '+str( X_coloumns[i])+':')
    list_of_variables.append(val)



prediction = predict(list_of_variables)

code_random ='''
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators= estimators, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
'''
code_logistic ='''
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
'''
code_naive ='''
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''
code_svm ='''
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
'''
st.write(list_of_variables)
# st.write(type(list_of_variables[0]))
if model['random_acc']>model['lc_acc'] and model['random_acc']>model['naive_acc']:
    st.write(prediction['randomf_predict'])
    st.success('model chosen by ensemble learning==>random_forest')
    st.code(code_random,language='python')

elif model['lc_acc']>model['random_acc'] and model['lc_acc']>model['naive_acc']:
    st.write(prediction['logistic_predict'])
    st.success('model chosen by ensemble learning==>logistic')
    st.code(code_logistic,language='python')

elif model['naive_acc']>model['random_acc'] and model['naive_acc']>model['lc_acc']:
    st.write(prediction['naive_predict'])
    st.success('model chosen by ensemble learning==>naive_bayes')
    st.code(code_naive,language='python')

elif model['svm_acc']>model['random_acc'] and model['svm_acc']>model['lc_acc']:
    st.write(prediction['svm_predict'])
    st.success('model chosen by ensemble learning==>svm')
    st.code(code_svm,language='python')
