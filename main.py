import streamlit as st
import pandas as pd

from PIL import Image
# Loading Image using PIL
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

# from PIL import Image
# st.write('no code model builder')
# image = Image.open('dark.png')
# st.image(image)
# st.write('description')
st.title('_No code Model builder_')




# if 'file' not in st.session_state:
#     st.session_state['file'] = ""
st.markdown("takes <span style='color:pink'>data,</span>,makes model,lets you **<span style='color:violet'>test</span>**   it.", unsafe_allow_html=True)
st.divider()

st.caption('upload your dataset for classification here:')

classify = st.file_uploader('',type = ['csv','xlsx'])

st.divider()

st.caption('upload your dataset for regression here:')

regression = st.file_uploader(' ',type = ['csv','xlsx'])


    
if classify:
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = classify.name
    st.write("Filename: ", classify.name)
    st.session_state['file'] = pd.read_csv(classify)
    st.write(st.session_state['file'])



if regression:
    if 'uploaded_file_egression' not in st.session_state:
        st.session_state['uploaded_file_egression'] = regression.name
    st.write("Filename: ", regression.name)
    st.session_state['file_egression'] = pd.read_csv(regression)
    st.write(st.session_state['file_egression'])

# import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('background.png')  








































































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