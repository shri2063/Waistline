import base64

import streamlit as st
import numpy as np
from streamlit_cropper import st_cropper
from PIL import Image, ImageOps
from streamlit_chat import message
import os

from llm.llm_response import ai_introduction, run_llm, chatgpt_call_with_memory, messages
from llm.llm_sizing import find_sizing_category_for_issue
from models.box import Box
from sizing.crawler import build_t_shirt_key_points
from sizing.yolo_model_prediction import model_json_prediction_for_sizing_issue

st.set_option('deprecation.showfileUploaderEncoding', False)
from zipfile import ZipFile
from sizing.sizing_pre_processing import correct_class_for_sleeves, get_corner_coordinates_for_tshirt
from quality.roboflow_inference import model_img_prediction, generate_response_based_upon_result, \
    get_iou_input_and_iou_predicted, yolo_chirag

st.header("WaistLyne")
# st.session_state.widget = ''
i = 45

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "issue" not in st.session_state:
    st.session_state.issue = 'quality'
if "generated_issue" not in st.session_state:
    st.session_state.generated_issue = ''
GENERATED_ISSUE: str = ''
# Upload an image and set some options for demo purposes

img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'], key="img_file")
img_folder = st.sidebar.file_uploader(label = "Upload image collection for sizing", type = "zip", key = "zipfile")
model = st.sidebar.radio(label="Select Model", options=["blue", "green"], key="model")
defect = st.sidebar.radio(label="Select defect", options=["quality", "sizing"], key="defect")
selected_folder = st.sidebar.selectbox("Select a folder: ", ["clean_tshirts", "black_tshirt", "green_tshirt"])
check_images = st.sidebar.button(label = "Check Images")
download_images = st.sidebar.button(label = "Download Images")
message(ai_introduction, key=i.__str__())

def submit(elseif=None):
    with st.spinner("Generating response...."):
        print("Session State " + str(st.session_state))
        cust_query = st.session_state.widget
        generated_response = run_llm(query=cust_query)
        global  GENERATED_ISSUE
        st.session_state.generated_issue = str(generated_response)
        print("Generated Issue", GENERATED_ISSUE)
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
    with open("predict/tmp.zip", "wb") as f:
        f.write(img_folder.read())
    # Extract all contents of zip folder to a temporary folder
    with  ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall("predict")
    st.success("Folder uploaded and extracted successfully")


IMAGE_CHECKED = False
if check_images:
    # Display the list of clean_tshirts in the uploaded folder
    print(selected_folder)
    directory = "samples/" + selected_folder
    image_files = [f for f in os.listdir(directory )]
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
    IMAGE_CHECKED = True
    #st.write("List of clean_tshirts in  the uploaded folder:")
    #print(image_files)
    #st.write(image_files[0])

if download_images:
    directory = "samples/" + selected_folder
    image_files = [f for f in os.listdir(directory)]
    zip_file_name = f"{selected_folder}_images.zip"

    with st.spinner(f"Creating {zip_file_name}.... "):
        st.write("Downloading...")
        with ZipFile(zip_file_name, 'w') as zipf:
            for image_file in image_files:
                image_path = os.path.join(directory, image_file)
                zipf.write(image_path, os.path.basename(image_file))
        with open(zip_file_name, "rb") as f:
            zip_contents = f.read()

        # Encode the zip file as base64
        zip_b64 = base64.b64encode(zip_contents).decode()
        href = f'<a href="data:application/zip;base64,{zip_b64}" download="{zip_file_name}">Click here to download</a>'
        st.markdown(href, unsafe_allow_html=True)


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
        box_color="yellow",
        aspect_ratio=(1, 1),
        return_type="box",
        stroke_width=2
    )
    CHEST = []
    SHOULDER = []
    TSHIRT = []

    if st.button('Submit'):

        if st.session_state.issue == "sizing":
            directory = "predict/images"
            image_files = [f for f in os.listdir(directory)]
            for image_file in image_files:
                print(image_file)
                #img = Image.open(os.path.join(directory, image_file))
                #raw_image = np.asarray(img).astype('uint8')
                #bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                #cv2.imwrite('sizing_img.jpg', bgr_image)
                predictions = model_json_prediction_for_sizing_issue(os.path.join(directory, image_file))
                predictions = get_corner_coordinates_for_tshirt(predictions)
                corrected_predictions = correct_class_for_sleeves(predictions)
                # print("Corrected Predictions")
                # print(corrected_predictions)
                try:
                    chest_length, shoulder_length, tshirt_length = build_t_shirt_key_points(corrected_predictions)
                    CHEST.append(chest_length)
                    SHOULDER.append(shoulder_length)
                    TSHIRT.append(tshirt_length)
                except Exception as e:
                    print(e.__str__())

            print(f"chest: " + str(sum(CHEST) / len(CHEST)))
            print(f"shoulder: " + str(sum(SHOULDER) / len(SHOULDER)))
            print(f"tshirt: " + str(sum(TSHIRT) / len(TSHIRT)))
            st.write('chest: ' + str(sum(CHEST) / len(CHEST)) + 'shoulder: ' +  str(sum(SHOULDER) / len(SHOULDER)) + "tshirt: " + str(sum(TSHIRT) / len(TSHIRT)))

        if st.session_state.issue == "quality":
            print("Session State: " + str(st.session_state))
            st.session_state["user_prompt_history"] = []
            st.session_state["chat_answers_history"] = []
            st.write('We are working on your query. Please wait.')

            #raw_image = np.asarray(img).astype('uint8')
            left, top, width, height = tuple(map(int, rect.values()))
            input_box = Box(
                x=left,
                y=top,
                width=width,
                height=height
            )
            bgr_image = img.convert("RGB")
            bgr_image.save("quality/quality_img.jpg")
            model = yolo_chirag()
            iou_input, iou_predicted = get_iou_input_and_iou_predicted(model, input_box)
            result, generated_response = generate_response_based_upon_result(iou_input, iou_predicted)
            message(generated_response, key=i.__str__())

    if st.button('Retry'):
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answers_history"] = []
        st.write('We are working on your query. Please wait.')

        raw_image = np.asarray(img).astype('uint8')
        left, top, width, height = tuple(map(int, rect.values()))
        input_box = Box(
            x=left,
            y=top,
            width=width,
            height=height
        )
        bgr_image = img.convert("RGB")
        bgr_image.save("quality/quality_img.jpg")
        model = yolo_chirag()
        iou_input, iou_predicted = get_iou_input_and_iou_predicted(model, input_box)
        result, generated_response = generate_response_based_upon_result(iou_input, iou_predicted)
        if result == True:
            generated_response = "Apologies for my earlier reply. " + generated_response
        message(generated_response, key=i.__str__())

if st.button('Summarize'):
    #messages.append({'role': 'user', 'content': 'Please summarize the discussion till now in format and structure ,so I can directly pass it as a prefix for my next prompt with no other context required  '})
    #response = chatgpt_call_with_memory(messages)
    # Split the string based on "Issue:"
    #GENERATED_ISSUE = "Thanks . I understand that you find tshirt chest size too long, but can you exaplain in detail if tshirt chest is wider or longer than you expected. Noting down the Issue:Sizing:chest size is longer than expected "


    print(st.session_state.generated_issue)
    split_result = str(st.session_state.generated_issue).split("Noting down the Issue")
    print(split_result)
    issue = ''
    # Check if there's a second part (index 1) after the split
    if len(split_result) > 1:
        # Extract the string after "Issue:"
        issue = split_result[1].strip()

    print("Issue", issue)
    response = find_sizing_category_for_issue(issue)
    st.write(response)




if st.button('Check result'):
    if model == "blue":
        model = yolo_chirag()
    else:
        model = yolo_chirag()

    st.write('We are working on your query. Please wait.')
    predicted_image_file = model_img_prediction(model, "quality/quality_img.jpg")
    predicted_image = Image.open(predicted_image_file)
    predicted_image = np.asarray(predicted_image).astype('uint8')
    st.image(Image.fromarray(predicted_image), caption='Predicted Image')

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                              st.session_state["user_prompt_history"]):
        i = i + 1
        message(user_query, is_user=True, key=i.__str__())
        i = i + 1
        message(generated_response, key=i.__str__())
