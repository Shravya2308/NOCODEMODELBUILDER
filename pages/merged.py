import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score






if 'file' in st.session_state:
    st.write(st.session_state['file'])

if 'var1' not in st.session_state:
    st.session_state['var1'] = ""

if  'mean' not in st.session_state:
    st.session_state['mean'] = ""

if 'input1' not in st.session_state:
    st.session_state['input'] = ""


# if 'var2' not in st.session_state:
#     st.session_state['var2'] = ""
n_estimators_o = st.number_input('no of estimators: ',min_value=5)
if n_estimators_o:
    st.session_state['var1'] = int(n_estimators_o)
    st.write(st.session_state['var1'])

df = st.session_state['file']
# for col in df.columns:
#     st.write(col)  



# st.write(X)
# st.write(y)


regret = st.button('Regret')

classify = st.button('Classify')
if regret:
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


    poly_regressor = PolynomialFeatures(degree = 4)
    x_poly = poly_regressor.fit_transform(X)
    lin_regressor_2 = LinearRegression()
    lin_regressor_2.fit(x_poly,y)



# st.write(lc_acc)     
    y = y.reshape(len(y),1)
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, y)

    X_coloumns  = list(df.iloc[:, :-1].columns.values)
    st.write(X_coloumns)
    a = len(X_coloumns) 
    predict = st.number_input('Insert a number')
    # list_of_variables = []
    # for i in range(a):
    #     var = i
    #     val = st.number_input('Enter your '+str( X_coloumns[i])+':')
    #     list_of_variables.append(val)

    svr = sc_y.inverse_transform(regressor.predict(sc_X.transform([[predict]])).reshape(1, -1))
    poly = lin_regressor_2.predict(poly_regressor.fit_transform([[predict]]))

    st.write((svr+poly)/2)
    if 'regret' in st.session_state:
        st.session_state['var1'] = (svr+poly)/2
        st.write(st.session_state['regret'])


if classify:
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

