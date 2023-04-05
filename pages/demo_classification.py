import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.classification import classifier
from models.classification import predict





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


X_coloumns  = list(df.iloc[:, :-1].columns.values)
st.write(X_coloumns)
a = len(X_coloumns) 
list_of_variables = []
for i in range(a):
    var = i
    val = st.number_input('Enter your '+str( X_coloumns[i])+':')
    list_of_variables.append(val)


prediction = predict(list_of_variables)

st.write(list_of_variables)
# st.write(type(list_of_variables[0]))
if model['random_acc']>model['lc_acc']:
    st.write(prediction['randomf_predict'])
    st.write('model chosen by ensemble learning==>random_forest')
else:
    st.write(prediction['logistic_predict'])
    st.write('model chosen by ensemble learning==>logistic')
