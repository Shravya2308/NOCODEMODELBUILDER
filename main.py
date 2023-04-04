import streamlit as st
import pandas as pd


from PIL import Image
st.write('no code model builder')
image = Image.open('dark.png')
st.image(image)
st.write('description')

# page_bg_img = '''
# <style>
# body {
# background-image: url("ackground.jpg");
# background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)

# # if 'default' not in st.session_state:
# #     st.session_state['default'] = ""
if 'file' not in st.session_state:
    st.session_state['file'] = ""

upload_file = st.file_uploader(label='upload your csv or excel file here',type = ['csv','xlsx'])



if upload_file:
    st.session_state['file'] = pd.read_csv(upload_file)
    st.write(st.session_state['file'])










































































# from PIL import Image
# import os

# st.write('no code model builder')
# image = Image.open('dark.png')
# st.image(image, caption='Sunrise by the mountains')

# st.write('idhar description likhenge')
# # st.info(__doc__)
# uploaded_file = st.file_uploader(label='upload your csv or eexcel file here',type = ['csv','xlsx'])
# process = st.button('process')
# default_csv_options = ["diabetes dataset","Social network dataset"]
# option_selected = st.selectbox("Choose a default csv file instead",options = default_csv_options )
# show = st.button('View Dataset')
# if process:
#     if uploaded_file is not None:
#         try:
#             data = pd.read_csv(uploaded_file)
#             st.write(data)
#             if show:
#                st.write(data)
#         except Exception as e:
#             print(e)
#             data = pd.read_excel(uploaded_file)

# if option_selected == "diabetes dataset":
#     data = pd.read_csv('diabetes.csv')
#     if show:
#      st.write(data)
# elif option_selected == "Social network dataset":
#     data= pd.read_csv('Social_Network_Ads.csv')
#     if show:
#      st.write(data)

    


# show_file = st.empty()
# if uploaded_file is not None:

#        print('File is empty')
# else:
#     st.write('your csv file is NULL. please upload a different file.')