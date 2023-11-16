#from roboflow import Roboflow
from pydantic import BaseModel, Field

import streamlit as st

import json
import streamlit as st
import  numpy as np
from roboflow import Roboflow

import numpy as np



from rf_sizing_pre_processing import get_gradient_quadrant_for_contour


def yolo_chirag():
    rf = Roboflow(api_key="jPnk3SftEgcEmCcfhN0F")
    project = rf.workspace("chirag-s3e7s").project("tshirt-evfwv")
    return project.version(4).model








# infer on a local image
# print(model.images("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.images("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.images("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


def model_img_prediction(model, filename: str) -> str:
    model.predict(filename, confidence=4, overlap=30).save("prediction.jpg")
    return "prediction.jpg"







def model_json_prediction(model, filename: str) -> list[Box]:
    model_prediction = model.predict(filename, confidence=4, overlap=30).json()
    #print(model_prediction)

    # Parse and validate the JSON data
    predictions_data = PredictionsData.parse_obj(model_prediction)

    # Access the extracted objects
    predictions = predictions_data.predictions
    box_objects = []
    for prediction in predictions:
        box = Box(
            x = prediction.x,
            y = prediction.y,
            width=prediction.width,
            height=prediction.height
        )
        #print(box)
        box_objects.append(box)
    return box_objects


def calculate_iou(input_box: Box, predicted_box: Box):

    x_intersection = max(input_box.x, predicted_box.x)
    y_intersection = max(input_box.y, predicted_box.y)
    x2_intersection = min(input_box.x + input_box.width, predicted_box.x + predicted_box.width)
    y2_intersection = min(input_box.y + input_box.height, predicted_box.y + predicted_box.height)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2_intersection - x_intersection) * max(0, y2_intersection - y_intersection)

    # Calculate the areas of both boxes
    area_box_input = input_box.width * input_box.height
    area_box_predicted = predicted_box.width * predicted_box.height

    # Calculate the Union area
    union_area = area_box_input + area_box_predicted - intersection_area

    # Calculate the IoU score
    iou_input_box = intersection_area/area_box_input
    iou_predicted_box =intersection_area/area_box_predicted
    return iou_input_box,iou_predicted_box


def generate_response_based_upon_result(iou_input: float, iou_predicted: float) :
    if iou_input > 0.15 and iou_predicted > 0.15:
        return True, "I can clearly see that there is a defect in the form of stain in the area you have highlighted. " \
                     "I have accepted your request and necessary actions will be taken"
    elif iou_input < 0.15 and iou_predicted > 0.15:
        return False, "Sorry! I am not able to detect defect in the area you have selected. I think the area you have selected is quite broad. " \
                      " Can you please reduce the area selected and  make it focus  precisely cover only the affected part"
    elif iou_input > 0.15 and iou_predicted < 0.15:
        return False, "Sorry! I am not able to detect defect in the area you have selected. I think the area you have selected is quite narrow. " \
                      " Can you please increase  the area selected and make it focus  precisely cover only the affected part"
    elif iou_input < 0.15 and iou_predicted < 0.15:
        return False, "Sorry! I am not able to detect defect in the area you have selected. " \
                      " Can you please readjust  the area selected and make it focus  precisely cover only the affected part"

def get_iou_input_and_iou_predicted(model, input_box) :
    f_iou_input = 0.0
    f_iou_predicted = 0.0
    box_objects: list[Box] = model_json_prediction(model, 'scaled_cropped_img.jpg')
    for predicted_box in box_objects:
        print(
            f"predicted box x: {predicted_box.x}, y: {predicted_box.y}, width: {predicted_box.width}, height: {predicted_box.height}")
        print(
            f"input box x: {input_box.x}, y: {input_box.y}, width: {input_box.width}, height: {input_box.height}")
        iou_input, iou_predicted = calculate_iou(input_box, predicted_box)
        if iou_input > f_iou_input and iou_predicted > f_iou_predicted:
            f_iou_input = iou_input
            f_iou_predicted = iou_predicted
        elif iou_input > f_iou_input and iou_predicted < f_iou_predicted:
            if f_iou_predicted < 0.15:
                f_iou_input = iou_input
                f_iou_predicted = iou_predicted
        elif iou_input < f_iou_input and iou_predicted > f_iou_predicted:
            if f_iou_input < 0.15:
                f_iou_input = iou_input
                f_iou_predicted = iou_predicted

        print("IoU Input: " + str(iou_input) + "  IoU Predicted: " + str(iou_predicted))
    return f_iou_input,f_iou_predicted




