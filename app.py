import base64
import webbrowser
import streamlit as st
import numpy as np
from streamlit_cropper import st_cropper
from PIL import Image, ImageOps
from streamlit_chat import message
import os
import shutil
from llm.llm_response import ai_introduction, run_llm, chatgpt_call_with_memory, chatgpt_call
from llm.llm_sizing import generate_sizing_category_for_issue, generate_response_based_upon_sizing_calculations
from models.box import Box
from models.t_shirt_size_chart import t_shirt_size_chart_ratio, t_shirt_size_chart_length
from sizing.crawler import build_t_shirt_key_points, get_ratios_for_tshirt, get_actual_length_for_tshirt
from sizing.yolo_model_prediction import model_json_prediction_for_sizing_issue

st.set_option('deprecation.showfileUploaderEncoding', False)
from zipfile import ZipFile
from sizing.sizing_pre_processing import correct_class_for_sleeves, get_corner_coordinates_for_tshirt
from quality.roboflow_inference import model_img_prediction, generate_response_based_upon_result, \
    get_iou_input_and_iou_predicted, yolo_chirag

st.header("WaistLyne v1.9")
# st.session_state.widget = ''
i = 45

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "llm_response" not in st.session_state:
    st.session_state["llm_response"] = []
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "issue_category" not in st.session_state:
    st.session_state.issue_category = ''
if "generated_issue" not in st.session_state:
    st.session_state.generated_issue = ''
if "sizing_category" not in st.session_state:
    st.session_state.sizing_category = ''
if "sizing_fist_ref" not in st.session_state:
    st.session_state.sizing_fist_ref = False
if "t_shirt_size" not in st.session_state:
    st.session_state.t_shirt_size = ''
st.set_option('deprecation.showfileUploaderEncoding', False)

img_file = None

# Upload an image and set some options for demo purposes

# img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'], key="img_file")
img_folder = st.sidebar.file_uploader(label="Upload image collection for sizing", type="zip", key="zipfile")
model = st.sidebar.radio(label="Select Model", options=["blue", "green"], key="model")
defect = st.sidebar.radio(label="Select defect", options=["quality", "sizing"], key="defect")
selected_folder = st.sidebar.selectbox("Select a folder: ", ["clean_tshirts", "black_tshirt", "green_tshirt"])
check_images = st.sidebar.button(label="Check Images")
download_images = st.sidebar.button(label="Download Images")
# Initial Message
message(ai_introduction, key=i.__str__())

# if st.session_state["chat_answers_history"]:
#    for generated_response, user_query in zip(st.session_state["chat_answers_history"],
#                                              st.session_state["user_prompt_history"]):
#        i = i + 1
#        message(user_query, is_user=True, key=i.__str__())
#       i = i + 1
#       message(generated_response, key=i.__str__())

if st.session_state["chat_messages"]:
    for chat_message in st.session_state["chat_messages"]:
        i = i + 1

        if chat_message["is_user"]:
            message(chat_message["message"], is_user=True, key=i.__str__())
        else:
            message(chat_message["message"], key=i.__str__())

#st.session_state.issue_category = 'sizing'
#st.session_state.sizing_fist_ref = False
if st.button("Open url in new tab"):
    st.markdown(f'<a href="https://drive.google.com/drive/folders/1aZ6fcShbc9pwZimKK1ndX9q5pitOKzS" target="_blank">Click here to open the URL in a new tab</a>',
                unsafe_allow_html=True)
    #webbrowser.open_new_tab("https://drive.google.com/drive/folders/1aZ6fcShbc9pwZimKK1ndX9q5pitOKzSq")
if st.session_state.issue_category == 'sizing' and st.session_state.sizing_fist_ref == False:
    sample_image = Image.open("sizing/sample_sizing_image.jpg")
    st.image(sample_image, caption="Sample Image", width=300)
    sizing_img = st.file_uploader(label='Upload Image of your tshirt', type=['png', 'jpg'], key="img_file")
    st.session_state.t_shirt_size = st.text_input("Please enter size  of your tshirt in Capital. Example: L")

    if sizing_img:
        sizing_img = Image.open(sizing_img)
        sizing_img.save('sizing/sizing_img.jpg')

    if st.button('Submit Image'):
        if st.session_state.t_shirt_size not in ('XS', 'S', 'M', 'L', 'XL', '2XL', '3XL', '4XL'):
            st.write("Tshirt size is not entered properly. Please select between XS,S,M,L,XL,2XL,3XL,4XL ")
        elif sizing_img is None:
            st.write("please upload an image")
        else:
            st.write("We are working on our request. Please wait")

            try:
                predictions = model_json_prediction_for_sizing_issue('sizing/sizing_img.jpg')
                predictions = get_corner_coordinates_for_tshirt(predictions)
                corrected_predictions = correct_class_for_sleeves(predictions)
                context = 'Here is the information available for a tshirt regarding its sizing.Please comprehend this information to answer the issue in query. ' \

                tshirt_size = st.session_state.t_shirt_size
                CbyS, CbyL, SbyL = get_ratios_for_tshirt(corrected_predictions)
                if t_shirt_size_chart_ratio[tshirt_size]['C/L'][0] <= CbyL <= \
                        t_shirt_size_chart_ratio[tshirt_size]['C/L'][1]:
                    context += f"Chest by Length ratio {CbyL} appears to be within range {t_shirt_size_chart_ratio[tshirt_size]['C/L'][0]} and " \
                               f"{t_shirt_size_chart_ratio[tshirt_size]['C/L'][1]}. That means size of chest to " \
                               f"Tshirt length appears to be correct"
                elif CbyL < t_shirt_size_chart_ratio[tshirt_size]['C/L'][0]:
                    context += (
                        f"Chest by Length ratio {CbyL} appears to be lesser than  range {t_shirt_size_chart_ratio[tshirt_size]['C/L'][0]} and "
                        f"{t_shirt_size_chart_ratio[tshirt_size]['C/L'][1]}. ")
                        #f"That means either chest size is smaller than expected in size {tshirt_size}"
                        #f"or if below Shoulder by Length is also less than expected in size {tshirt_size} then Tshirt length size is greater than expected in size {tshirt_size} ")
                elif CbyL > t_shirt_size_chart_ratio[tshirt_size]['C/L'][1]:
                    context += (
                        f"Chest by Length ratio {CbyL} appears to be greater than  range {t_shirt_size_chart_ratio[tshirt_size]['C/L'][0]} and "
                        f"{t_shirt_size_chart_ratio[tshirt_size]['C/L'][1]}. ")
                        #f"That means either chest size is greater than expected in size {tshirt_size}"
                        #f"or if below Shoulder by Length is also greater than expected in size {tshirt_size} then tshirt length size is smaller than expected in size {tshirt_size} ")

                if t_shirt_size_chart_ratio[tshirt_size]['S/L'][0] <= SbyL <= \
                        t_shirt_size_chart_ratio[tshirt_size]['S/L'][1]:
                    context += f" Shoulder by  Length ratio {SbyL} appears to be within range {t_shirt_size_chart_ratio[tshirt_size]['S/L'][0]} and " \
                               f"{t_shirt_size_chart_ratio[tshirt_size]['S/L'][1]}. Although we cannot conclude tshit size {tshirt_size} from this particular " \
                               f"information, it can be said that Shoulder to tshirt ratio appears correct"
                elif SbyL < t_shirt_size_chart_ratio[tshirt_size]['S/L'][0]:
                    context += (
                        f" Shoulder by Length ratio {SbyL} appears to be lesser than  range {t_shirt_size_chart_ratio[tshirt_size]['S/L'][0]} and " \
                        f"{t_shirt_size_chart_ratio[tshirt_size]['S/L'][1]}. "
                        f"If both Chest By length ratio is also less than expected then Tshirt length is less than expected" )
                        #f"That means either shoulder size is smaller than expected in size {tshirt_size}"
                        #f"or Tshirt length size is greater than expected in size {tshirt_size} ")
                elif SbyL > t_shirt_size_chart_ratio[tshirt_size]['S/L'][1]:
                    context += (
                        f" Shoulder by Length ratio {SbyL} appears to be greater than  range {t_shirt_size_chart_ratio[tshirt_size]['S/L'][0]} and "
                        f"{t_shirt_size_chart_ratio[tshirt_size]['S/L'][1]}. "
                        f"If both Chest By length ratio is also more  than expected then Tshirt length is more than expected")
                        #f"That means either shoulder  size is greater than expected in size {tshirt_size}"
                        #f"or Tshirt length size is smaller than expected in size {tshirt_size} ")

                if t_shirt_size_chart_ratio[tshirt_size]['C/S'][0] <= CbyS <= \
                        t_shirt_size_chart_ratio[tshirt_size]['C/S'][1]:
                    context += f" Chest by Shoulder ratio {CbyS} appears to be within range {t_shirt_size_chart_ratio[tshirt_size]['C/S'][0]} and " \
                               f"{t_shirt_size_chart_ratio[tshirt_size]['C/S'][1]}. Size of chest to  shoulder appears to be correct and if  above " \
                               f"Chest by  Length ratio is also within expected range that means the size of Tshirt {tshirt_size} seems to be correct" \

                elif CbyS < t_shirt_size_chart_ratio[tshirt_size]['C/S'][0]:
                    context += (
                        f" Chest by Shoulder ratio {CbyS} appears to be lesser than  range {t_shirt_size_chart_ratio[tshirt_size]['C/S'][0]} and " \
                        f"{t_shirt_size_chart_ratio[tshirt_size]['C/S'][1]}. Size of Tshirt {tshirt_size} does not seem  to be correct. "
                        f"If given above Chest by Length ratio is also smaller than expected that  means  chest size is smaller "
                        f"than expected in size {tshirt_size}"
                        f"or if  given above Shoulder  by Length ratio is also greater than expected then shoulder size is greater than expected in size {tshirt_size} ")
                elif CbyS > t_shirt_size_chart_ratio[tshirt_size]['C/S'][1]:
                    context += (
                        f" Chest by Shoulder ratio {CbyS} appears to be greater than  range {t_shirt_size_chart_ratio[tshirt_size]['C/S'][0]} and "
                        f"{t_shirt_size_chart_ratio[tshirt_size]['C/S'][1]}. Size of Tshirt {tshirt_size} does not seem  to be correct. "
                        f"If given above Chest by Length ratio is also greater that  means "
                        f" chest size is greater than expected in size {tshirt_size}"
                        f"or if  given above  Shoulder  by Length ratio is also smaller than expected then shoulder size is smaller than expected in size {tshirt_size} ")


                st.write(context)
                st.write(st.session_state.generated_issue)
                # st.write(st.session_state.generated_issue)

                response = generate_response_based_upon_sizing_calculations(st.session_state.generated_issue, context)
                st.session_state.sizing_fist_ref = True
                st.write(response)
                response = response + (" However, if you want me to get precise calculation. I would like to "
                                       "have multiple images of the same tshirt in a zip folder.Also I need to have "
                                       "reference of any mobile to be placed on the image to be used as a reference ")
                st.write(response)
                i = i + 1
                st.session_state["chat_messages"].append({"is_user": False, "message": response})
                message(response, key=i.__str__())
            except Exception as e:
                st.write(
                    "I am really sorry! There seems some problem with above tshirt image. Can you please upload another image following above instruction")
                print(e.__str__())

print("sizing_fist_ref", st.session_state.sizing_fist_ref)

if st.session_state.issue_category == 'sizing' and st.session_state.sizing_fist_ref == True:
    sample_image = Image.open("sizing/sample_sizing_img_w_mobile.jpeg")
    st.image(sample_image, caption="Sample Image", width=300)
    sizing_folder = st.file_uploader(label="Upload images zip  for sizing", type="zip", key="zipfile1")
    CHEST = []
    SHOULDER = []
    TSHIRT = []
    if st.button('Submit Folder'):
        if sizing_folder is None:
            st.write("Please upload zip folder with images of tshirt as described above")
        else:
            st.write("We are working on our request. Please wait")
            with open("sizing/tmp.zip", "wb") as f:
                if os.path.exists("sizing/predict"):
                    shutil.rmtree("sizing/predict")
                f.write(sizing_folder.read())
                # Extract all contents of zip folder to a temporary folder
                with  ZipFile("sizing/tmp.zip", "r") as zip_ref:
                    zip_ref.extractall("sizing/predict")
                    st.success("Folder uploaded and extracted successfully")
                    subdirectories = [d for d in os.listdir("sizing/predict") if os.path.isdir(os.path.join("sizing/predict", d))]

                    image_files = [f for f in os.listdir(os.path.join("sizing/predict", subdirectories[0]))]
                    for image_file in image_files:
                        print(image_file)
                        predictions = model_json_prediction_for_sizing_issue(os.path.join("sizing/predict", subdirectories[0], image_file))
                        predictions = get_corner_coordinates_for_tshirt(predictions)
                        corrected_predictions = correct_class_for_sleeves(predictions)

                        try:
                            chest_length, shoulder_length, tshirt_length = get_actual_length_for_tshirt(
                                corrected_predictions)
                            CHEST.append(chest_length)
                            SHOULDER.append(shoulder_length)
                            TSHIRT.append(tshirt_length)
                            chest_avg = round((sum(CHEST) / len(CHEST))*1.05, 2)
                            print("Chest: ", chest_avg)
                            shoulder_avg = round((sum(SHOULDER) / len(SHOULDER))*1.05, 2)
                            print("Shoulder: ", shoulder_avg)
                            tshirt_avg = round((sum(TSHIRT) / len(TSHIRT))*1.07, 2)
                            print("Tshirt: ", tshirt_avg)
                        except ValueError as e:
                            st.write(e.__str__())
                        except Exception as e:
                            print(e.__str__())
                    tshirt_size = st.session_state.t_shirt_size
                    chest_min, chest_max = t_shirt_size_chart_length[tshirt_size]['chest'][0], \
                        t_shirt_size_chart_length[tshirt_size]['chest'][1]
                    shoulder_min, shoulder_max = t_shirt_size_chart_length[tshirt_size]['shoulder'][0], \
                        t_shirt_size_chart_length[tshirt_size]['shoulder'][1]
                    tshirt_min, tshirt_max = t_shirt_size_chart_length[tshirt_size]['tshirt'][0], \
                        t_shirt_size_chart_length[tshirt_size]['tshirt'][1]

                    context = (
                        'Here is the information available for a tshirt regarding its sizing. Also please in the response provide'
                        'as much as numerical lengths along with min and  max values possible')
                    if chest_min <= chest_avg <= chest_max:
                        context += f"Tshirt's Chest length {chest_avg} cm is within expected range of {chest_min} cm  and {chest_max}cm"
                    elif chest_avg < chest_min:
                        context += f"Tshirt's Chest length {chest_avg}  cm is lesser than expected range of {chest_min} cm"
                    elif chest_avg > chest_max:
                        context += f"Tshirt's Chest length {chest_avg} cm is greater than expected range of {chest_max} cm"

                    if shoulder_min <= shoulder_avg <= shoulder_max:
                        context += f"Tshirt's Shoulder length {shoulder_avg} cm is within expected range  of {shoulder_min} cm and {shoulder_max} cm"
                    elif shoulder_avg < shoulder_min:
                        context += f"Tshirt's Shoulder length {shoulder_avg} cm is lesser than expected range of {shoulder_min}cm"
                    elif shoulder_avg > shoulder_max:
                        context += f"Tshirt's Shoulder length {shoulder_avg} cm is greater than expected range of {shoulder_max}cm"

                    if tshirt_min <= tshirt_avg <= tshirt_max:
                        context += f"Tshirt's  length {tshirt_avg} cm is within expected range  of {tshirt_min} cm and {tshirt_max} cm"
                    elif tshirt_avg < tshirt_min:
                        context += f"Tshirt's  length {tshirt_avg} cm is lesser than expected range of {tshirt_min}cm"
                    elif tshirt_avg > tshirt_max:
                        context += f"Tshirt's Shoulder length {tshirt_avg} cm is greater than expected range of {tshirt_max}cm"

                    response = generate_response_based_upon_sizing_calculations(st.session_state.generated_issue,
                                                                                context)
                    i = i + 1
                    message(response, key=i.__str__())
                    st.session_state["chat_messages"].append({"is_user": False, "message": response})
                    st.session_state.issue_category = ""


def submit(elseif=None):
    with st.spinner("Generating response...."):
        print("Session State " + str(st.session_state))
        cust_query = st.session_state.widget
        print(cust_query)
        st.write(cust_query)
        st.session_state["chat_messages"].append({"is_user": True, "message": cust_query})
        generated_response = run_llm(cust_query, fetch_conversation_till_now())
        st.write(generated_response)

        processed_response = run_change_detector(cust_query, str(generated_response))
        print(processed_response)
        st.session_state["chat_messages"].append({"is_user": False, "message": processed_response})


st.text_input("Prompt", key="widget", placeholder="Enter your prompt here ..", on_change=submit)


def run_change_detector(cust_query, generated_response):
    if "Sizing:" in generated_response:
        st.session_state.issue_category = 'sizing'
        st.session_state.sizing_fist_ref = False
        print("Session State: " + str(st.session_state.issue_category))
        st.write("Generated response: " + generated_response)
        category = get_sizing_category_for_issue(generated_response)
        st.session_state.sizing_category = category
        print("Category", category)
        st.write("Category", category)
        if category == "I don't know":
            st.session_state.issue_category = ''

            return ("I am really sorry. I understand you have issue in sizing related to tshirt."
                    "But I am not able to clearly understand the exact issue. If you don't mind can you"
                    "explain a little more in detail")
        #elif category == "Category B":
            #st.session_state.issue_category = ''

           # return ("I am really sorry. I understand you have issue in sizing related to tshirt."
                   # "But currently I am not able to measure sleeves and neck length. hence, i won't be able to solve your query. Hopefully, soon i will "
                    #"be able to measure the same")
        else:

            return generated_response + (". To provide additional assistance, please take a photo of "
                                         "the T-shirt and enter its size in the input box below. "
                                         "Ensure that you turn the T-shirt inside out and hold "
                                         "your phone over the top of the T-shirt, similar t"
                                         "o the example image attached. If you don't have "
                                         "a T-shirt available, you can select a relevant "
                                         "T-shirt with an issue from our gallery,"
                                         " as this is a demonstration.")

    elif "Quality:" in str(generated_response):
        st.session_state.issue_category = 'quality'
        print("Session State: " + str(st.session_state.issue_category))
        return generated_response
    else:
        st.session_state.issue_category = ""
        st.session_state["user_prompt_history"].append(cust_query)
        st.session_state["llm_response"].append(generated_response)
        return generated_response


def get_sizing_category_for_issue(gernerated_response):
    split_result = str(gernerated_response).split("Noting down")
    print(split_result)
    issue = ''
    # Check if there's a second part (index 1) after the split
    if len(split_result) > 1:
        # Extract the string after "Issue:"
        issue = split_result[1].strip()
        st.session_state.generated_issue = issue

    print("Issue", issue)
    st.write("Issue " + issue)
    return generate_sizing_category_for_issue(issue)


def fetch_conversation_till_now():
    conversation = []
    for generated_response, user_query in zip(st.session_state["llm_response"],
                                              st.session_state["user_prompt_history"]):
        new_query = {"query": user_query, "answer": generated_response}
        conversation.append(new_query)
        return conversation


if img_file is not None:

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

        if st.session_state.issue_category == "sizing":
            directory = "predict/images"
            image_files = [f for f in os.listdir(directory)]
            for image_file in image_files:
                print(image_file)
                # img = Image.open(os.path.join(directory, image_file))
                # raw_image = np.asarray(img).astype('uint8')
                # bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite('sizing_img.jpg', bgr_image)
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
            st.write('chest: ' + str(sum(CHEST) / len(CHEST)) + 'shoulder: ' + str(
                sum(SHOULDER) / len(SHOULDER)) + "tshirt: " + str(sum(TSHIRT) / len(TSHIRT)))

        if st.session_state.issue_category == "quality":
            print("Session State: " + str(st.session_state))
            st.session_state["user_prompt_history"] = []
            st.session_state["chat_answers_history"] = []
            st.write('We are working on your query. Please wait.')

            # raw_image = np.asarray(img).astype('uint8')
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

    # if st.button('Check result'):
    # if model == "blue":
    # model = yolo_chirag()
    # else:
    # model = yolo_chirag()

    st.write('We are working on your query. Please wait.')
    predicted_image_file = model_img_prediction(model, "quality/quality_img.jpg")
    predicted_image = Image.open(predicted_image_file)
    predicted_image = np.asarray(predicted_image).astype('uint8')
    st.image(Image.fromarray(predicted_image), caption='Predicted Image')

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
    image_files = [f for f in os.listdir(directory)]
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
    IMAGE_CHECKED = True
    # st.write("List of clean_tshirts in  the uploaded folder:")
    # print(image_files)
    # st.write(image_files[0])

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
