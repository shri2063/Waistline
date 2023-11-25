
from roboflow import Roboflow

from models.PredictionItem import PredictionsData
from models.box import Box


def yolo_chirag():
    rf = Roboflow(api_key="jPnk3SftEgcEmCcfhN0F")
    project = rf.workspace("chirag-s3e7s").project("tshirt-evfwv")
    return project.version(4).model

def yolo_venkatesh():
    rf = Roboflow(api_key="DQ9wqnpIdZMAPHBHp6qt")
    project = rf.workspace("venkatesh-nbix7").project("t-shirt-with-defect")
    return project.version(1).model

def yolo_rahul():
    rf = Roboflow(api_key="tltyisBDbiERQLoieJpR")
    project = rf.workspace("rahul-3uiyz").project("t-shirt-defect-detection")
    return project.version(1).model








# infer on a local image
# print(model.clean_tshirts("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.clean_tshirts("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.clean_tshirts("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


def model_img_prediction(model, filename: str) -> str:
    model.predict(filename, confidence=10, overlap=30).save("quality/predict.jpg")
    return "quality/predict.jpg"







def model_json_prediction(model, filename: str, confidence) -> list[Box]:
    model_prediction = model.predict(filename, confidence=confidence, overlap=30).json()
    print(model_prediction)

    #predictions_data = PredictionsData(predictions=[])
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
        return True, "I can clearly see that there is a defect in the area you have highlighted. " \
                     "I will accept your request"
    elif iou_input < 0.15 and iou_predicted > 0.15:
        return False, "I apologize, but I'm unable to detect a defect in the area you've chosen. It seems that the selected area is quite extensive. " \
                      "Could you please narrow down the selected area to precisely cover only the affected part? Alternatively, " \
                      "you can take another image focused on the specific area of concern, using a camera flash for better clarity."
    elif iou_input > 0.15 and iou_predicted < 0.15:
        return False, "I apologize, but I'm unable to detect a defect in the area you've chosen. It appears that the selected area is too narrow. " \
                      "Could you please increase the selected area, ensuring it precisely covers only the affected part? " \
                      "This adjustment will help in a more accurate assessment." \
                      " Alternatively, you can take another image focused on the specific area of concern, using a camera flash for better clarity."
    elif iou_input < 0.15 and iou_predicted < 0.15:

        return False, "I apologize, but I couldn't detect a defect in the area you selected. " \
                      "Please click on Retry if you want me try one more time. " \
                      "Alternatively, could you please readjust the selected area, " \
                      "ensuring it focuses precisely on the affected part? This adjustment will assist in a more accurate analysis." \
                      " Additionally, you can take another image focused on the specific area of concern, using a camera flash for better clarity."

def get_iou_input_and_iou_predicted(model, input_box,confidence ) :
    f_iou_input = 0.0
    f_iou_predicted = 0.0
    box_objects: list[Box] = model_json_prediction(model, "quality/quality_img.jpg", confidence)
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




