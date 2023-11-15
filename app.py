import base64

import streamlit as st
import numpy as np
from streamlit_cropper import st_cropper
from PIL import Image, ImageOps
from llm_response import run_llm
from streamlit_chat import message
import os
st.set_option('deprecation.showfileUploaderEncoding', False)




st.header("Waist Lyne Functioning Tour")
# st.session_state.widget = ''
i = 45

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "issue" not in st.session_state:
    st.session_state.issue = 'sizing'
# Upload an image and set some options for demo purposes

img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'], key="img_file")
img_folder = st.sidebar.file_uploader(label = "Upload image collection for sizing", type = "zip", key = "zipfile")
model = st.sidebar.radio(label="Select Model", options=["blue", "green"], key=4)
selected_folder = st.sidebar.selectbox("Select a folder: ", ["images"])
check_images = st.sidebar.button(label = "Check Images")
download_images = st.sidebar.button(label = "Download Images")

def submit(elseif=None):
    with st.spinner("Generating response...."):
        print("Session State " + str(st.session_state))
        cust_query = st.session_state.widget
        generated_response = run_llm(query=cust_query)
        print(generated_response)
        if "Sizing Issue" in str(generated_response):
            st.session_state.issue = 'sizing'
            print("Session State: " + str(st.session_state.issue))
        elif "Quality Issue" in str(generated_response):
            st.session_state.issue = 'quality'
            print("Session State: " + str(st.session_state.issue))

        st.session_state["user_prompt_history"].append(cust_query)
        st.session_state["chat_answers_history"].append(generated_response)

st.text_input("Prompt", key="widget", placeholder="Enter your prompt here ..", on_change=submit)

st.set_option('deprecation.showfileUploaderEncoding', False)


if img_folder:
    with open("temp.zip", "wb") as f:
        f.write(img_folder.read())
    # Extract all contents of zip folder to a temporary folder
    #with  ZipFile("temp.zip", "r") as zip_ref:
        #zip_ref.extractall("predict")
    st.success("Folder uploaded and extracted successfully")


IMAGE_CHECKED = False
if check_images:
    # Display the list of images in the uploaded folder
    print(selected_folder)
    directory = "samples/" + selected_folder
    image_files = [f for f in os.listdir(directory )]
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
    IMAGE_CHECKED = True
    #st.write("List of images in  the uploaded folder:")
    #print(image_files)
    #st.write(image_files[0])

if download_images:
    directory = "samples/" + selected_folder
    image_files = [f for f in os.listdir(directory)]
    zip_file_name = f"{selected_folder}_images.zip"

    with st.spinner(f"Creating {zip_file_name}.... "):
        st.write("Downloading...")
        #with ZipFile(zip_file_name, 'w') as zipf:
            #for image_file in image_files:
               # image_path = os.path.join(directory, image_file)
               # zipf.write(image_path, os.path.basename(image_file))
       # with open(zip_file_name, "rb") as f:
            #zip_contents = f.read()

        # Encode the zip file as base64
        #zip_b64 = base64.b64encode(zip_contents).decode()
        #href = f'<a href="data:application/zip;base64,{zip_b64}" download="{zip_file_name}">Click here to download</a>'
        #st.markdown(href, unsafe_allow_html=True)


if img_file:
    img = Image.open(img_file)
    img = ImageOps.exif_transpose(img)
    width, height = img.size

    if width > 200.0:
        new_height = height / width * 200.0
        new_width = 200.0
        img = img.resize((int(new_width), int(new_height)))

    elif height > 200.0:
        new_width = width / height * 200.0
        new_height = 200.0
        img = img.resize((int(new_width), int(new_height)))



    rect = st_cropper(
        img,
        realtime_update=True,
        box_color="blue",
        aspect_ratio=(1, 1),
        return_type="box",
        stroke_width=2
    )
    CHEST = []
    SHOULDER = []
    TSHIRT = []
    if st.button('Submit'):
        print("Session State: " + str(st.session_state))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                              st.session_state["user_prompt_history"]):
        i = i + 1
        message(user_query, is_user=True, key=i.__str__())
        i = i + 1
        message(generated_response, key=i.__str__())
