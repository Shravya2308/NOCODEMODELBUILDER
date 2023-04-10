import streamlit as st



# name= input('Please enter the name of the receipient: ')
# url= input('please enter the URL: ')
my_name= st.session_state['uploaded_file']
mail_body= f"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
df = pd.read_csv('{st.session_state['uploaded_file']}')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)
    
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lc_model = LogisticRegression()
lc_model.fit(X_train, y_train)
y_pred_l = lc_model.predict(X_test)
    


lc_acc = accuracy_score(y_test, y_pred_l)

randomforest = RandomForestClassifier(n_estimators= {st.session_state['n_estimators']}, criterion = 'entropy', random_state = 0)
randomforest.fit(X_train, y_train)
y_pred_c = randomforest.predict(X_test)
random_acc = accuracy_score(y_test, y_pred_c)
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_n = naive_bayes.predict(X_test)
naive_acc = accuracy_score(y_test, y_pred_n)

svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(X_train, y_train)
y_pred_s = svm.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_s)



"""
st.code(mail_body,language='python')