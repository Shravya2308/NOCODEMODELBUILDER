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

if 'var1' not in st.session_state:
    st.session_state['var1'] = ""

# if 'var2' not in st.session_state:
#     st.session_state['var2'] = ""
n_estimators_o = st.number_input('no of estimators: ',min_value=5)
if n_estimators_o:
    st.session_state['var1'] = int(n_estimators_o)
    st.write(st.session_state['var1'])

df = st.session_state['file']
# for col in df.columns:
#     st.write(col)  
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values 


model = classifier(df,n_estimators_o)

# st.write(X)
# st.write(y)


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
if model['classifier_acc']>model['lc_acc']:
    st.write(prediction['randomf_predict'])
    st.write('model chosen by ensemble learning==>random_forest')
else:
    st.write(prediction['logistic_predict'])
    st.write('model chosen by ensemble learning==>logistic')
