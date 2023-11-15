#from roboflow import Roboflow
from pydantic import BaseModel, Field
import csv
import streamlit as st
import matplotlib.pyplot as plt
import json
import streamlit as st
import  numpy as np
from roboflow import Roboflow
from ultralytics import YOLO
import numpy as np
from skimage import measure
import json
from pydantic import BaseModel, Field



from rf_sizing_pre_processing import get_gradient_quadrant_for_contour
t_shirt_segments = {
    "4": "t_shirt",
    "2": "neck",
    "3": "right_sleeve",
    "1": "mobile",
    "0": "left_sleeve"
}


class Box:
    def __init__(self, x:float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height







    def create_box(self):
        box = Box(
            x= self.x,
            y = self.y,
            width = self.width,
            height=self.height
        )
        return box


class ImageData(BaseModel):
    width: str
    height: str
class Coordinate(BaseModel):
    x: float
    y: float
class CornerCoordinate(BaseModel):
    left_coordinate: tuple = None
    right_coordinate: tuple = None
    top_coordinate: tuple = None
    bottom_coordinate: tuple = None
class PredictionItem(BaseModel):
    x: float = Field(description="Left")
    y: float = Field(description="Top")
    width: float = Field(description="Width")
    height: float = Field(description="Height")
    confidence: float
    class_: str = 'bla'  # 'class' is a reserved keyword, so we use 'class_' instead
    class_id: int
    points: list[Coordinate]
    corner_coordinate: CornerCoordinate = None
class PredictionsData(BaseModel):
    predictions: list[PredictionItem]
    #image: ImageData

#def yolo_tushar():
    # rf = Roboflow(api_key="fDIRWlltWtFjYt8t20bU")
    #project = rf.workspace("tushar-x68o7").project("t-shirt-ix6cg")
    #return project.version(1).model

def get_prediction_using_YOLO():
    model = YOLO("best.pt")
    results = model.predict(source = "sizing_img.jpg", conf = 0.25)
    masks = results[0].masks.cpu()
    #print(masks.data)
    for mask in masks.data:
        #print(mask)
        contours = measure.find_contours(np.array(mask), 0.5)
        contour_coordinates = [contour for contour in contours]
        points = [{"x": x, "y": y} for y, x in contour_coordinates[0]]
        #print(points)



    boxes = results[0].boxes
    #print(np.array(boxes[0].xyxy)[0,0])
    classes = np.array(boxes.cls)
    cnf = boxes.conf
    predictions_data = PredictionsData(predictions = [])



    i = 0
    for mask in masks.data:
        contours = measure.find_contours(np.array(mask), 0.5)
        contour_coordinates = [contour for contour in contours]
        points = [{"x": x, "y": y} for y, x in contour_coordinates[0]]
        prediction_item = PredictionItem(x=np.array(boxes[i].xyxy)[0,0], y=np.array(boxes[i].xyxy)[0,1], width=np.array(boxes[i].xyxy)[0,2],
                                         height=np.array(boxes[i].xyxy)[0,3],
                                         confidence=cnf[i] , class_id= classes[i],
                                         points=points,  class_ = "bla")
        #print(prediction_item)
        predictions_data.predictions.append(prediction_item)
        i = i + 1
        # print(i)
    # Convert the JSON object to a JSON string
    # Convert the PredictionsData object to a dictionary
    predictions_data_dict = predictions_data.dict()
    json_string = json.dumps(predictions_data_dict, indent=4)
    file_path = 't_shirt.txt'
    with open(file_path, 'w') as file:
        file.write(json_string)
    return predictions_data





#def yolo_chirag():
    # rf = Roboflow(api_key="jPnk3SftEgcEmCcfhN0F")
    #project = rf.workspace("chirag-s3e7s").project("tshirt-evfwv")
    #return project.version(4).model
    #print("changing to Tushar")







# infer on a local image
# print(model.images("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.images("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.images("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


def model_img_prediction(model, filename: str) -> str:
    model.predict(filename, confidence=4, overlap=30).save("prediction.jpg")
    return "prediction.jpg"

def model_json_prediction_for_sizing_issue(filename: str) -> any:
    #model_prediction = model.images(filename, confidence=20).json()
    #print(model_prediction)
    #file_path = "D:\Desktop\ice-breaker\intro-to-vector\data.txt"
    #with open(file_path, "w") as file:
       #json.dump(model_prediction, file)
    # Open the file in read mode and load the JSON data
   # with open(file_path, "r") as file:
        # model_prediction = json.load(file)

    #predictions_data = PredictionsData.parse_obj(model_prediction)
    predictions_data = get_prediction_using_YOLO()
    predictions = predictions_data.predictions
    csv_file_path = "points.csv"
    x = []
    y = []
    t_shirt_contour = []
    with open(csv_file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x-coordinate", "y-coordinate", "class"])
        for prediction in predictions:
            prediction.class_ = t_shirt_segments[str(prediction.class_id)]
            for point in prediction.points:
                csv_writer.writerow([point.x, point.y, prediction.class_])
                if prediction.class_ == "t_shirt":
                    x.append(int(point.x))
                    y.append(int(point.y))
                    t_shirt_contour.append((point.x, point.y))

    # Create a scatter plot
    t_shirt_contour = np.array(t_shirt_contour)
    print("Plotting scatter plot")
    plt.scatter(x, y)

    # Add labels and a title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')

    # Show the plot
    plt.show()
    plt.savefig('scatter_plot.png')


   # build_t_shirt_key_points(t_shirt_contour, "sizing_img.jpg")
    for prediction in predictions:
        sleeve_predictions = []
        if "sleeve" in prediction.class_:
            sleeve_contour = []
            for point in prediction.points:
                sleeve_contour.append((point.x, point.y))
            sleeve_contour = np.array(sleeve_contour)
            get_gradient_quadrant_for_contour(sleeve_contour)

    return predictions










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




