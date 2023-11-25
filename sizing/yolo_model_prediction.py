from skimage import measure
import json
from pydantic import BaseModel, Field
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import csv

from models.PredictionItem import PredictionsData, PredictionItem

t_shirt_segments = {
    "4": "t_shirt",
    "2": "neck",
    "3": "right_sleeve",
    "1": "mobile",
    "0": "left_sleeve"
}


class ImageData(BaseModel):
    width: str
    height: str


# def yolo_tushar():
# rf = Roboflow(api_key="fDIRWlltWtFjYt8t20bU")
# project = rf.workspace("tushar-x68o7").project("t-shirt-ix6cg")
# return project.version(1).model

def get_prediction_using_YOLO(filename: str):
    model = YOLO("trained_models/best_1.pt")
    results = model.predict(source=filename, conf=0.3)
    masks = results[0].masks.cpu()


    boxes = results[0].boxes
    #print(boxes)
    classes = np.array(boxes.cls)
    #print(classes)
    cnf = boxes.conf
    #print(cnf)
    predictions_data = PredictionsData(predictions=[])

    i = 0
    #print(masks.xy[0])
    for mask in masks.data:

        contours = measure.find_contours(np.array(mask), 0.5)
        contour_coordinates = [contour for contour in contours]
        points = [{"x": x, "y": y} for y, x in contour_coordinates[0]]




        prediction_item = PredictionItem(x=np.array(boxes[i].xyxy)[0, 0], y=np.array(boxes[i].xyxy)[0, 1],
                                         width=np.array(boxes[i].xyxy)[0, 2],
                                         height=np.array(boxes[i].xyxy)[0, 3],
                                         confidence=cnf[i], class_id=classes[i],
                                         points=points, class_="bla")
        # print(prediction_item)

        predictions_data.predictions.append(prediction_item)
        i = i + 1
    #neck_item_with_max_cnf = max((prediction_item for prediction_item in predictions_data.predictions if prediction_item.class_id == 2), key = lambda x: x.confidence,default=None)
    #predictions_data.predictions = [prediction_item for prediction_item in predictions_data.predictions if prediction_item.class_id != 2
    #                                 or(prediction_item.class_id == 2 and prediction_item.confidence == neck_item_with_max_cnf.confidence)]
    #for item in   predictions_data.predictions:
    #    print(item.class_id)
        #print(predictions_data)
        # print(i)
    # Convert the JSON object to a JSON string
    # Convert the PredictionsData object to a dictionary

    # file_path = 'sizing/t_shirt.txt'
    # with open(file_path, 'w') as file:
    # file.write(json_string)
    return predictions_data


def model_json_prediction_for_sizing_issue(filename: str) -> any:
    # model_prediction = model.clean_tshirts(filename, confidence=20).json()
    # print(model_prediction)
    # file_path = "D:\Desktop\ice-breaker\intro-to-vector\data.txt"
    # with open(file_path, "w") as file:
    # json.dump(model_prediction, file)
    # Open the file in read mode and load the JSON data
    # with open(file_path, "r") as file:
    # model_prediction = json.load(file)

    # predictions_data = PredictionsData.parse_obj(model_prediction)
    predictions_data = get_prediction_using_YOLO(filename)
    predictions = predictions_data.predictions
    csv_file_path = "sizing\contour_points.csv"
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
                if prediction.class_ == "mobile":
                    x.append(int(point.x))
                    y.append(int(point.y))
                    t_shirt_contour.append((point.x, point.y))

                if "sleeve" in prediction.class_:
                    x.append(int(point.x))
                    y.append(int(point.y))
                    t_shirt_contour.append((point.x, point.y))

    print("Plotting scatter plot")
    plt.scatter(x, y)

    # Add labels and a title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')

    # Show the plot
    # plt.show()
    # plt.savefig('sizing\scatter_plot.png')

    # build_t_shirt_key_points(t_shirt_contour, "sizing_img.jpg")
    # for prediction in predictions:
    # sleeve_predictions = []
    # if "sleeve" in prediction.class_:
    # sleeve_contour = []
    # for point in prediction.points:
    # sleeve_contour.append((point.x, point.y))
    # sleeve_contour = np.array(sleeve_contour)
    # get_gradient_quadrant_for_contour(sleeve_contour)

    return predictions
