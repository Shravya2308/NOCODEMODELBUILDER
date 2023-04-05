import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import streamlit as st




dataset = pd.read_csv('50_Startups.csv')
dataset = dataset.drop(['State'], axis=1) 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)


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

st.write(ereg.score(X_test,y_test))