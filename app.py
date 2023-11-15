import streamlit as st
import numpy as np
from streamlit_cropper import st_cropper
from PIL import Image
from llm_response import run_llm
from streamlit_chat import message

st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Cropper Demo")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
stroke_width = st.sidebar.number_input(label="Box Thickness", value=3, step=1)

aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

return_type_choice = st.sidebar.radio(label="Return type", options=["Cropped image", "Rect coords"])
return_type_dict = {
    "Cropped image": "image",
    "Rect coords": "box"
}
return_type = return_type_dict[return_type_choice]

st.header("Ice breaker Helper Bot")
# st.session_state.widget = ''
i = 45

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "issue" not in st.session_state:
    st.session_state.issue = 'sizing'


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

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    if return_type == 'box':
        rect = st_cropper(
            img,
            realtime_update=realtime_update,
            box_color=box_color,
            aspect_ratio=aspect_ratio,
            return_type=return_type,
            stroke_width=stroke_width
        )
        raw_image = np.asarray(img).astype('uint8')
        left, top, width, height = tuple(map(int, rect.values()))
        st.write(rect)
        masked_image = np.zeros(raw_image.shape, dtype='uint8')
        masked_image[top:top + height, left:left + width] = raw_image[top:top + height, left:left + width]
        st.image(Image.fromarray(masked_image), caption='masked image')
    else:
        # Get a cropped image from the frontend
        cropped_img = st_cropper(
            img,
            realtime_update=realtime_update,
            box_color=box_color,
            aspect_ratio=aspect_ratio,
            return_type=return_type,
            stroke_width=stroke_width
        )

        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150, 150))
        st.image(cropped_img)
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                              st.session_state["user_prompt_history"]):
        i = i + 1
        message(user_query, is_user=True, key=i.__str__())
        i = i + 1
        message(generated_response, key=i.__str__())
