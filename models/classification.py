
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

sc = StandardScaler()
lc_model = LogisticRegression()

def classifier(dataframe,estimators):
    df = dataframe
    # for col in df.columns:
    #     st.write(col)  
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values 



    # st.write(X)
    # st.write(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)
    
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
     #object
    lc_model.fit(X_train, y_train)
    y_pred_l = lc_model.predict(X_test)
    


    lc_acc = accuracy_score(y_test, y_pred_l)
    # st.write(lc_acc)     

    #RANDOM FOREST
    global randomforest
    randomforest = RandomForestClassifier(n_estimators= estimators, criterion = 'entropy', random_state = 0)
    randomforest.fit(X_train, y_train)
   
    y_pred_c = randomforest.predict(X_test)
    random_acc = accuracy_score(y_test, y_pred_c)
    return {'lc_acc':lc_acc , 'random_acc':random_acc}

def predict(list_of_variables):
    randomf_predict = randomforest.predict(sc.transform([list_of_variables]))
    logistic_predict =lc_model.predict(sc.transform([list_of_variables]))
    return {'randomf_predict':randomf_predict,'logistic_predict':logistic_predict}