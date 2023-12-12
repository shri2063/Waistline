import base64

import streamlit as st
import numpy as np
from streamlit_cropper import st_cropper
from PIL import Image, ImageOps
from streamlit_chat import message
import os
import shutil
from llm.llm_response import ai_introduction, run_llm, secret_key
from llm.llm_sizing import generate_sizing_category_for_issue, generate_response_based_upon_sizing_calculations
from models.box import Box
from models.t_shirt_size_chart import t_shirt_size_chart_ratio, t_shirt_size_chart_length
from sizing.crawler import build_t_shirt_key_points, get_ratios_for_tshirt, get_actual_length_for_tshirt
from sizing.sizing_response import  calculate_lengths_for_image, \
    get_context_based_upon_lengths
from sizing.yolo_model_prediction import model_json_prediction_for_sizing_issue


from zipfile import ZipFile
from sizing.sizing_pre_processing import correct_class_for_sleeves, get_corner_coordinates_for_tshirt
from quality.roboflow_inference import model_img_prediction, generate_response_based_upon_result, \
    get_iou_input_and_iou_predicted,  yolo_venkatesh, yolo_rahul

st.header("Waistline v1.12")
# st.session_state.widget = ''
i = 45

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "llm_response" not in st.session_state:
    st.session_state["llm_response"] = []
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "issue_category" not in st.session_state:
    st.session_state.issue_category = 'sizing'
if "generated_issue" not in st.session_state:
    st.session_state.generated_issue = ''
if "sizing_category" not in st.session_state:
    st.session_state.sizing_category = ''
if "sizing_fist_ref" not in st.session_state:
    st.session_state.sizing_fist_ref = False
if "context" not in st.session_state:
    st.session_state.context = ''
if "auth" not in st.session_state:
    st.session_state.auth = False

SECRET_KEYS = ["flexli", "Bewakoof.com"]
img_file = None

# Upload an image and set some options for demo purposes

# img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'], key="img_file")
# img_folder = st.sidebar.file_uploader(label="Upload image collection for sizing", type="zip", key="zipfile")
#model = st.sidebar.radio(label="Select Model", options=["blue", "green"], key="model")

MOBILE_LENGTH = 15.8
MOBILE_WIDTH = 7.3
# MOBILE_LENGTH = 1.0
# MOBILE_WIDTH = 1.0
# defect = st.sidebar.radio(label="Select defect", options=["quality", "sizing"], key="defect")
# selected_folder = st.sidebar.selectbox("Select a folder: ", ["clean_tshirts", "black_tshirt", "green_tshirt"])
# heck_images = st.sidebar.button(label="Check Images")
# ownload_images = st.sidebar.button(label="Download Images")
# Initial Message
if(st.session_state.auth):
    message(ai_introduction, key=i.__str__())
else:
    message(secret_key, key=i.__str__())

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


def is_image_file(filename):
    image_extensions = (".png", ".jpeg", ".jpg")
    return any(filename.endswith(ext) for ext in image_extensions)


#st.session_state.issue_category = 'sizing'
#st.session_state.sizing_fist_ref = False
if st.session_state.issue_category == 'sizing' and st.session_state.sizing_fist_ref == False and st.session_state.auth:

    st.write(
        "For this demonstration, if you don't have a relevant T-shirt, you can use our catalogue to fetch any tshirt image. Please click on Catalogue button below."
        " Afterward, please upload the chosen image to continue with the conversation.")
    if st.button("Our Catalogue"):
        st.markdown(
            f'<a href="https://drive.google.com/drive/folders/1aS2MKUiXdXOWOF_KAGoHb2zkifddQ4ah">Click here to open the URL in a new tab</a>',
            unsafe_allow_html=True)
    sizing_img = st.file_uploader(label='Upload Image of your tshirt', type=['png', 'jpg', 'zip'], key="img_file")

    st.write("In case you plan to use any of your tshirt. Please take an image of tshirt like below from a distance of approximate 3ft")
    sample_image = Image.open("sizing/sample_sizing_image.jpg")
    st.image(sample_image, caption="Sample Image", width=300)


    #if sizing_img:
        #st.write(sizing_img.type)

    if st.button('Submit Image'):


        if sizing_img is None:
            st.write("please upload an image")
        elif sizing_img.type == "image/jpeg" or sizing_img.type == "image/jpg" \
                or sizing_img.type == "image/png":
            st.success("Image file uploaded successfully!")
            st.write("We are working on our request. Please wait")
            sizing_img = Image.open(sizing_img)
            sizing_img.save('sizing/sizing_img.jpg')
            x = Image.open('sizing/sizing_img.jpg')
            width, height = x.size
            ratio = height / width
            #st.write("ratio", ratio)
            width_list = [width, 450.0, 550.0, 650.0]
            context = ''
            for width in width_list:
                try:
                    img = x.resize((int(width), int(ratio * width)))
                    print(img.size)
                    img_name = 'sizing/resize/sizing_img_' + str(width) + ".jpg"
                    img.save(img_name)
                    chest_length, shoulder_length, tshirt_length = calculate_lengths_for_image(img_name)
                    context = get_context_based_upon_lengths(chest_length, shoulder_length,
                                                             tshirt_length)
                    break

                except Exception as e:
                    print(e.__str__())


        elif sizing_img.name.endswith(".zip"):
            st.success("ZIP file uploaded successfully!")
            st.write(
                "We are working on our request. Please wait, it might take 2 to 3 minutes based upon size of folder")
            CHEST = []
            SHOULDER = []
            TSHIRT = []

            with open("sizing/tmp.zip", "wb") as f:
                if os.path.exists("sizing/predict"):
                    shutil.rmtree("sizing/predict")
                f.write(sizing_img.read())
                # Extract all contents of zip folder to a temporary folder
                with  ZipFile("sizing/tmp.zip", "r") as zip_ref:
                    zip_ref.extractall("sizing/predict")
                    st.success("Folder uploaded and extracted successfully")
                    subdirectories = [d for d in os.listdir("sizing/predict") if
                                      os.path.isdir(os.path.join("sizing/predict", d))]
                    image_files = [f for f in os.listdir(os.path.join("sizing/predict", subdirectories[0]))]
                    for image_file in image_files:
                        x = Image.open(os.path.join("sizing/predict", subdirectories[0], image_file))
                        width, height = x.size
                        ratio = height / width
                        print(ratio)
                        width_list = [width, 450.0, 550.0, 650.0]

                        for width in width_list:
                            try:

                                print(image_file)
                                img = x.resize((int(width), int(ratio * width)))
                                img_name = 'sizing/resize/sizing_img_' + str(width) + ".jpg"
                                img.save(img_name)
                                predictions = model_json_prediction_for_sizing_issue(img_name)
                                predictions = get_corner_coordinates_for_tshirt(predictions)
                                corrected_predictions = correct_class_for_sleeves(predictions)
                                chest_length, shoulder_length, tshirt_length = get_actual_length_for_tshirt(
                                    corrected_predictions)
                                CHEST.append(chest_length)
                                SHOULDER.append(shoulder_length)
                                TSHIRT.append(tshirt_length)

                                break


                            except ValueError as e:

                                st.write(e.__str__())

                            except Exception as e:

                                print(e.__str__())


                    chest_avg = round((sum(CHEST) / len(CHEST)) * 1.0, 2)
                    # print("Chest", CHEST)
                    print("Chest Avg: ", chest_avg)
                    shoulder_avg = round((sum(SHOULDER) / len(SHOULDER)) * 1.0, 2)
                    print("Shoulder Avg: ", shoulder_avg)
                    tshirt_avg = round((sum(TSHIRT) / len(TSHIRT)) * 1.0, 2)
                    print("Tshirt Avg: ", tshirt_avg )
                    context = get_context_based_upon_lengths(chest_avg, shoulder_avg,
                                                             tshirt_avg, st.session_state.t_shirt_size)

        else:
            st.warning("Unsupported file type. Please upload an image (PNG or JPEG) or a ZIP file.")

        if context != '':
            st.session_state.context = context
            #st.write(st.session_state.generated_issue)
            response = generate_response_based_upon_sizing_calculations(st.session_state.generated_issue,
                                                                                    context)
            # st.session_state.sizing_fist_ref = True
            #st.write(response)
            i = i + 1
            st.session_state["chat_messages"].append({"is_user": False, "message": response})
            message(response, key=i.__str__())
            #st.session_state.issue_category = ""

    if st.button("Retry"):
        context = st.session_state.context
        st.write("retrying")
        print("Retrying.........")
        response = generate_response_based_upon_sizing_calculations(st.session_state.generated_issue,
                                                                            context)
        # st.session_state.sizing_fist_ref = True
        #st.write(response)
        print("response")
        i = i + 1
        st.session_state["chat_messages"].append({"is_user": False, "message": response})
        message(response, key=i.__str__())
        #st.session_state.issue_category = ""







print("sizing_fist_ref", st.session_state.sizing_fist_ref)
#st.session_state.issue_category = 'sizing'
#st.session_state.sizing_fist_ref = True

if st.session_state.issue_category == 'quality':
    st.write(
        "As this is a demonstration, if you don't have relevent image feel free to download either a defective or non-defective image by clicking the link below. "
        "Please upload the chosen image to proceed with the conversation")
    if st.button("Check Tshirt Quality images"):
        st.markdown(
            f'<a href="https://drive.google.com/drive/folders/138eWBPMLCzJMWXpSEapQOwC_z6YPaG2A" target="_blank">Click here to open the URL in a new tab</a>',
            unsafe_allow_html=True)
    img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg'], key="img_file")

    if img_file is not None:
        img = Image.open(img_file)
        img = ImageOps.exif_transpose(img)
        width, height = img.size

        if width > 700.0:
            new_height = height / width * 700.0
            new_width = 700.0
            img = img.resize((int(new_width), int(new_height)))

        elif height > 700.0:
            new_width = width / height * 700.0
            new_height = 700.0
            img = img.resize((int(new_width), int(new_height)))

        rect = st_cropper(
            img,
            realtime_update=True,
            box_color="yellow",
            aspect_ratio=(1, 1),
            return_type="box",
            stroke_width=2
        )
    if st.button('Submit Image'):
        if img_file is None:
            st.write("Please upload image of tshirt as described above")
        else:

            print("Session State: " + str(st.session_state))
            st.write('We are working on your query. Please wait.')

            left, top, width, height = tuple(map(int, rect.values()))
            input_box = Box(
                x=left,
                y=top,
                width=width,
                height=height)
            bgr_image = img.convert("RGB")
            bgr_image.save("quality/quality_img.jpg")
            model = yolo_venkatesh()
            iou_input, iou_predicted = get_iou_input_and_iou_predicted(model, input_box, 10)
            result, generated_response = generate_response_based_upon_result(iou_input, iou_predicted)
            i = i + 1
            message(generated_response, key=i.__str__())

    if st.button('Retry'):
        st.write('We are working on your query. Please wait.')
        left, top, width, height = tuple(map(int, rect.values()))
        input_box = Box(
            x=left,
            y=top,
            width=width,
            height=height
        )
        bgr_image = img.convert("RGB")
        bgr_image.save("quality/quality_img.jpg")
        model = yolo_rahul()
        iou_input, iou_predicted = get_iou_input_and_iou_predicted(model, input_box, 10)
        result, generated_response = generate_response_based_upon_result(iou_input, iou_predicted)
        if result:
            generated_response = "Apologies for my earlier reply. " + generated_response
            i = i + 1
            message(generated_response, key=i.__str__())
        if not result:
            iou_input, iou_predicted = get_iou_input_and_iou_predicted(yolo_venkatesh(), input_box, 5)
            result, generated_response = generate_response_based_upon_result(iou_input, iou_predicted)
            if result:
                generated_response = "Apologies for my earlier reply. " + generated_response
                i = i + 1
                message(generated_response, key=i.__str__())
            else:
                i = i + 1
                message(generated_response, key=i.__str__())

    if st.button('Check result'):
        model = yolo_venkatesh()


        st.write('We are working on your query. Please wait.')
        predicted_image_file = model_img_prediction(model, "quality/quality_img.jpg")
        predicted_image = Image.open(predicted_image_file)
        predicted_image = np.asarray(predicted_image).astype('uint8')
        st.image(Image.fromarray(predicted_image), caption='Predicted Image', width=300)


def submit(elseif=None):

    with st.spinner("Generating response...."):
        if st.session_state.auth == False :
            if (st.session_state.widget  in SECRET_KEYS):
                st.session_state.auth = True
                return
            else:
                st.write("Sorry your key is not correct. Please try again")
                return


        #print("Session State " + str(st.session_state))
        st.session_state.issue_category = ""
        cust_query = st.session_state.widget
        #print(cust_query)
        #st.write(cust_query)
        st.session_state["chat_messages"].append({"is_user": True, "message": cust_query})
        generated_response = run_llm(cust_query, fetch_conversation_till_now())
        #st.write(generated_response)

        processed_response = run_change_detector(cust_query, str(generated_response))
        print(processed_response)
        st.session_state["chat_messages"].append({"is_user": False, "message": processed_response})


st.text_input("Prompt", key="widget", placeholder="Enter your prompt here ..", on_change=submit)


def run_change_detector(cust_query, generated_response):
    if "Sizing:" in generated_response:
        st.session_state.issue_category = 'sizing'
        st.session_state.sizing_fist_ref = False
        print("Session State: " + str(st.session_state.issue_category))
        #st.write("Generated response: " + generated_response)
        category = get_sizing_category_for_issue(generated_response)
        st.session_state.sizing_category = category
        print("Category", category)
        #st.write("Category", category)
        if category == "I don't know":
            st.session_state.issue_category = ''

            return ("I am really sorry. I understand you have issue in sizing related to tshirt."
                    "But I am not able to clearly understand the exact issue. If you don't mind can you"
                    "explain a little more in detail")
        # elif category == "Category B":
        # st.session_state.issue_category = ''

        # return ("I am really sorry. I understand you have issue in sizing related to tshirt."
        # "But currently I am not able to measure sleeves and neck length. hence, i won't be able to solve your query. Hopefully, soon i will "
        # "be able to measure the same")
        else:

            return generated_response + (". To provide additional assistance, please take a photo of "
                                         "the T-shirt and enter its size in the input box below. "
                                         "Ensure that you turn the T-shirt inside out and hold "
                                         "your phone over the top of the T-shirt at approximately 3ft height, similar t"
                                         "o the example image attached.")

    elif "Quality:" in str(generated_response):
        st.session_state.issue_category = 'quality'
        print("Session State: " + str(st.session_state.issue_category))
        if "mismatch" in str(generated_response):
            st.session_state.issue_category = ""
            return (
                "I sincerely apologize for the inconvenience. I acknowledge that there is a quality mismatch issue with the T-shirt."
                " However, at the moment, I am unable to address any quality mismatch concerns as I do not have access to the original images of the T-shirt. "
                "If the quality issue pertains to stains, holes, or similar issues, I am more than willing to assist in resolving it.")
        else:
            return generated_response + (
                ". To provide additional assistance, please take a closer photo of concerned portion of the tshirt. "
                "Please ensure that concerned part is clearly visible.")

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
    #st.write("Issue " + issue)
    return generate_sizing_category_for_issue(issue)


def fetch_conversation_till_now():
    conversation = []
    for generated_response, user_query in zip(st.session_state["llm_response"],
                                              st.session_state["user_prompt_history"]):
        new_query = {"query": user_query, "answer": generated_response}
        conversation.append(new_query)
        return conversation


# if st.session_state.issue_category == 'sizing' and st.session_state.sizing_fist_ref == True:
#     # st.write("below tshirt image is just for demonstration how to capture image")
#     # sample_image = Image.open("sizing/sample_sizing_img_w_mobile.jpeg")
#     # st.image(sample_image, caption="Sample Image", width=300)
#     st.write(
#         "For this demonstration, in case you don't have the relevant image, you can download an image or a zip folder containing any "
#         "T-shirt with the expected size by clicking the link below. Kindly upload the selected image to continue with the conversation.")
#     if st.button("Check Tshirts Zip folder"):
#         st.markdown(
#             f'<a href="https://drive.google.com/drive/folders/1cCB1LK94ELvz4CQfSifjlELKhJUZJBG6">Click here to open the URL in a new tab</a>',
#             unsafe_allow_html=True)
#     sizing_folder = st.file_uploader(label="Upload images zip  for sizing", type="zip", key="zipfile1")
#
#     if st.button('Submit Folder'):
#         if sizing_folder is None:
#             st.write("Please upload zip folder with images of tshirt as described above")
#         else:
#             st.write(
#                 "We are working on our request. Please wait, it might take 2 to 3 minutes based upon size of folder")
#             CHEST = []
#             SHOULDER = []
#             TSHIRT = []
#             P_X = []
#             P_Y = []
#             with open("sizing/tmp.zip", "wb") as f:
#                 if os.path.exists("sizing/predict"):
#                     shutil.rmtree("sizing/predict")
#                 f.write(sizing_folder.read())
#                 # Extract all contents of zip folder to a temporary folder
#                 with  ZipFile("sizing/tmp.zip", "r") as zip_ref:
#                     zip_ref.extractall("sizing/predict")
#                     st.success("Folder uploaded and extracted successfully")
#                     subdirectories = [d for d in os.listdir("sizing/predict") if
#                                       os.path.isdir(os.path.join("sizing/predict", d))]
#                     image_files = [f for f in os.listdir(os.path.join("sizing/predict", subdirectories[0]))]
#                     for image_file in image_files:
#                         x = Image.open(os.path.join("sizing/predict", subdirectories[0], image_file))
#                         width, height = x.size
#                         ratio = height / width
#                         print(ratio)
#                         width_list = [width, 450.0, 550.0, 650.0]
#
#                         for width in width_list:
#                             try:
#
#                                 print(image_file)
#                                 img = x.resize((int(width), int(ratio * width)))
#                                 img_name = 'sizing/resize/sizing_img_' + str(width) + ".jpg"
#                                 img.save(img_name)
#                                 predictions = model_json_prediction_for_sizing_issue(img_name)
#                                 predictions = get_corner_coordinates_for_tshirt(predictions)
#                                 corrected_predictions = correct_class_for_sleeves(predictions)
#                                 chest_length, shoulder_length, tshirt_length, p_x, p_y = get_actual_length_for_tshirt(
#                                     corrected_predictions)
#                                 CHEST.append(chest_length)
#                                 SHOULDER.append(shoulder_length)
#                                 TSHIRT.append(tshirt_length)
#                                 P_X.append(p_x)
#                                 P_Y.append(p_y)
#
#                                 break
#
#
#                             except ValueError as e:
#
#                                 st.write(e.__str__())
#
#                             except Exception as e:
#
#                                 print(e.__str__())
#
#                     p_x_avg = round((sum(P_X) / len(P_X)) / MOBILE_WIDTH, 2)
#                     print("P_X: ", p_x_avg)
#                     # print(P_X)
#                     # print(P_Y)
#                     p_y_avg = round((sum(P_Y) / len(P_Y)) / MOBILE_LENGTH, 2)
#                     print("P_Y: ", p_y_avg)
#                     chest_avg = round((sum(CHEST) / len(CHEST)) * 1.0, 2)
#                     # print("Chest", CHEST)
#                     print("Chest: ", chest_avg / p_x_avg)
#                     shoulder_avg = round((sum(SHOULDER) / len(SHOULDER)) * 1.0, 2)
#                     print("Shoulder: ", shoulder_avg / p_x_avg)
#                     tshirt_avg = round((sum(TSHIRT) / len(TSHIRT)) * 1.0, 2)
#                     print("Tshirt: ", tshirt_avg / p_y_avg)
#
#                     tshirt_size = st.session_state.t_shirt_size
#                     chest_min, chest_max = t_shirt_size_chart_length[tshirt_size]['chest'][0], \
#                         t_shirt_size_chart_length[tshirt_size]['chest'][1]
#                     shoulder_min, shoulder_max = t_shirt_size_chart_length[tshirt_size]['shoulder'][0], \
#                         t_shirt_size_chart_length[tshirt_size]['shoulder'][1]
#                     tshirt_min, tshirt_max = t_shirt_size_chart_length[tshirt_size]['tshirt'][0], \
#                         t_shirt_size_chart_length[tshirt_size]['tshirt'][1]
#
#                     context = (
#                         'Here is the information available for a tshirt regarding its sizing. Also please in the response provide'
#                         'as much as numerical lengths along with min and  max values possible. ')
#                     if chest_min <= chest_avg <= chest_max:
#                         context += f"Tshirt's Chest length {chest_avg} cm is within expected range of {chest_min} cm  and {chest_max}cm. "
#                     elif chest_avg < chest_min:
#                         context += f"Tshirt's Chest length {chest_avg}  cm is lesser than expected range of {chest_min} cm. "
#                     elif chest_avg > chest_max:
#                         context += f"Tshirt's Chest length {chest_avg} cm is greater than expected range of {chest_max} cm. "
#
#                     if shoulder_min <= shoulder_avg <= shoulder_max:
#                         context += f"Tshirt's Shoulder length {shoulder_avg} cm is within expected range  of {shoulder_min} cm and {shoulder_max} cm. "
#                     elif shoulder_avg < shoulder_min:
#                         context += f"Tshirt's Shoulder length {shoulder_avg} cm is lesser than expected range of {shoulder_min}cm. "
#                     elif shoulder_avg > shoulder_max:
#                         context += f"Tshirt's Shoulder length {shoulder_avg} cm is greater than expected range of {shoulder_max}cm. "
#
#                     if tshirt_min <= tshirt_avg <= tshirt_max:
#                         context += f"Tshirt's  length {tshirt_avg} cm is within expected range  of {tshirt_min} cm and {tshirt_max} cm. "
#                     elif tshirt_avg < tshirt_min:
#                         context += f"Tshirt's  length {tshirt_avg} cm is lesser than expected range of {tshirt_min}cm. "
#                     elif tshirt_avg > tshirt_max:
#                         context += f"Tshirt's Shoulder length {tshirt_avg} cm is greater than expected range of {tshirt_max}cm. "
#
#                     st.write(context)
#                     # response = generate_response_based_upon_sizing_calculations(st.session_state.generated_issue,
#                     #                                                           context)
#                     i = i + 1
#                     message(response, key=i.__str__())
#                     st.session_state["chat_messages"].append({"is_user": False, "message": response})
#                     st.session_state.issue_category = ""
#
#
#
#
#
#
#
#
#
#
#
#
#




#
# if img_folder:
#     with open("predict/tmp.zip", "wb") as f:
#         f.write(img_folder.read())
#     # Extract all contents of zip folder to a temporary folder
#     with  ZipFile("temp.zip", "r") as zip_ref:
#         zip_ref.extractall("predict")
#     st.success("Folder uploaded and extracted successfully")
#
# IMAGE_CHECKED = False
# if check_images:
#     # Display the list of clean_tshirts in the uploaded folder
#     print(selected_folder)
#     directory = "samples/" + selected_folder
#     image_files = [f for f in os.listdir(directory)]
#     for image_file in image_files:
#         image_path = os.path.join(directory, image_file)
#         image = Image.open(image_path)
#         st.image(image, use_column_width=True)
#     IMAGE_CHECKED = True
#     # st.write("List of clean_tshirts in  the uploaded folder:")
#     # print(image_files)
#     # st.write(image_files[0])
#
# if download_images:
#     directory = "samples/" + selected_folder
#     image_files = [f for f in os.listdir(directory)]
#     zip_file_name = f"{selected_folder}_images.zip"
#
#     with st.spinner(f"Creating {zip_file_name}.... "):
#         st.write("Downloading...")
#         with ZipFile(zip_file_name, 'w') as zipf:
#             for image_file in image_files:
#                 image_path = os.path.join(directory, image_file)
#                 zipf.write(image_path, os.path.basename(image_file))
#         with open(zip_file_name, "rb") as f:
#             zip_contents = f.read()
#
#         # Encode the zip file as base64
#         zip_b64 = base64.b64encode(zip_contents).decode()
#         href = f'<a href="data:application/zip;base64,{zip_b64}" download="{zip_file_name}">Click here to download</a>'
#         st.markdown(href, unsafe_allow_html=True)
