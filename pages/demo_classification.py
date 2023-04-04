import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score






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



# st.write(X)
# st.write(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lc_model = LogisticRegression() #object
lc_model.fit(X_train, y_train)
y_pred_l = lc_model.predict(X_test)


lc_acc = accuracy_score(y_test, y_pred_l)
# st.write(lc_acc)     

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=st.session_state['var1'], criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred_c = classifier.predict(X_test)
classifier_acc = accuracy_score(y_test, y_pred_c)
X_coloumns  = list(df.iloc[:, :-1].columns.values)
st.write(X_coloumns)
a = len(X_coloumns) 
list_of_variables = []
for i in range(a):
    var = i
    val = st.number_input('Enter your '+str( X_coloumns[i])+':')
    list_of_variables.append(val)




st.write(list_of_variables)
# st.write(type(list_of_variables[0]))
if classifier_acc>lc_acc:
    st.write(classifier.predict(sc.transform([list_of_variables])))
    st.write('model chosen by ensemble learning==>random_forest')
else:
    st.write(lc_model.predict(sc.transform([list_of_variables])))
    st.write('model chosen by ensemble learning==>logistic')
