import streamlit as st

def add():
    input_1 = st.number_input('Enter a number')
    input_2 = st.number_input('Enter another number')
    if st.button('click'):
        st.write(input_1+input_2)

st.button("main",on_click=add)