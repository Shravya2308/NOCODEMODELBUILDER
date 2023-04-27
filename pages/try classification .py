import streamlit as st
import pandas as pd
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




if 'file' in st.session_state:
    # st.write(st.session_state['file'])

    my_bar = st.progress(0)
    for percent_complete in range(80):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    if 'n_estimators' not in st.session_state:
        st.session_state['n_estimators'] = ""


    n_estimators_o = st.number_input('no of estimators: ',min_value=5)
    if n_estimators_o:
        st.session_state['n_estimators'] = int(n_estimators_o)
        st.write(st.session_state['n_estimators'])

    df = st.session_state['file']




    model = classifier(df,n_estimators_o)

    st.write('Logistic Regression',model['lc_acc'])
    st.write('Random Forest',model['random_acc'])
    st.write('Naive Bayes',model['naive_acc'])
    st.write('SVM',model['svm_acc'])

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


    if st.button('generate code'):
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
else:
    st.warning("no classification file selected")