import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image
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

# my_bar = st.progress(0)
# for percent_complete in range(80):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)


if 'file_egression' in st.session_state:
    my_bar = st.progress(0)
    for percent_complete in range(80):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    st.write(st.session_state['file_egression'])

    dataset = st.session_state['file_egression']

    # dataset = dataset.drop(['State'], axis=1) 
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    st.write(X)
    st.write(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    sc = StandardScaler()

    reg1 = GradientBoostingRegressor(random_state=1)
    reg2 = RandomForestRegressor(random_state=1)
    reg3 = LinearRegression()


    reg1.fit(X_train, y_train)
    # reg1.score(X_test,y_test)
    reg2.fit(X_train, y_train)
    # reg2.score(X_test,y_test)
    reg3.fit(X_train, y_train)
    # reg2.score(X_test,y_test)



    ereg = VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])
    ereg.fit(X_train, y_train)

    st.write(ereg.score(X_test,y_test))


    X_coloumns  = list(dataset.iloc[:, :-1].columns.values)
    st.write(X_coloumns)
    a = len(X_coloumns) 
    list_of_variables = []
    for i in range(a):
        var = i
        val = st.number_input('Enter your '+str( X_coloumns[i])+':')
        list_of_variables.append(val)

    st.write(ereg.predict([list_of_variables]))
    if st.button('generate code'):
        regression_code= st.session_state['uploaded_file_egression']
        mail_body_regression= f"""
        import pandas as pd
        import time
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import VotingRegressor
        from sklearn.preprocessing import StandardScaler

        dataset = pd.read_csv('{st.session_state['uploaded_file_egression']}')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        reg1 = GradientBoostingRegressor(random_state=1)
        reg2 = RandomForestRegressor(random_state=1)
        reg3 = LinearRegression()
        reg1.fit(X_train, y_train)
        # reg1.score(X_test,y_test)
        reg2.fit(X_train, y_train)
        # reg2.score(X_test,y_test)
        reg3.fit(X_train, y_train)
        # reg2.score(X_test,y_test)
        ereg = VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])
        ereg.fit(X_train, y_train)
        ereg.score(X_test,y_test)



        """
        st.code(mail_body_regression,language='python')
else:
    st.warning('no regression file selected')