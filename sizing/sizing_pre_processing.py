
import numpy as np
import matplotlib.pyplot as plt


from models.corner_coordinates import CornerCoordinate


def get_index_of_left_corner_pt(contour):
    index = 0
    for i in range(len(contour)):
        try:
            if i == len(contour) - 1:
                break
            if contour[i][0] < contour[index][0]:
                index = i
        except:
            1 == 1
    return index

def get_index_of_right_corner_pt(contour):
    index = 0
    for i in range(len(contour)):
        try:
            if i == len(contour) - 1:
                break
            if contour[i][0] > contour[index][0]:
                index = i
        except:
            1 == 1
    return index

def get_index_of_top_corner_pt(contour):
    index = 0
    for i in range(len(contour)):
        try:
            if i == len(contour) - 1:
                break
            if contour[i][1] < contour[index][1]:
                index = i
        except:
            1 == 1
    return index

def get_index_of_bottom_corner_pt(contour):
    index = 0
    for i in range(len(contour)):
        try:
            if i == len(contour) - 1:
                break
            if contour[i][1] > contour[index][1]:
                index = i
        except:
            1 == 1
    return index




def get_index_of_right_corner_pt(contour):
    index = 0
    for i in range(len(contour)):
        try:
            if i == len(contour) - 1:
                break
            if contour[i][0] > contour[index][0]:
                index = i
        except:
            1 == 1
    return index


def get_gradient_at_point(contour, index):
    gradient = 0.0
    try:
        if index < 5:
            return gradient
        if index > len(contour) - 6:
            return gradient
        prev_point = np.array(contour[index - 5])
        curr_point = np.array(contour[index])
        next_point = np.array(contour[index + 5])

        vector1 = curr_point - prev_point
        vector2 = curr_point - next_point

        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        gradient = np.arccos(dot_product / (magnitude1 * magnitude2))
        return gradient
    except:
        1 == 1


def get_corner_coordinates_for_tshirt(predictions):
    tshirt_prediction = [x for x in predictions if "t_shirt" in x.class_][0]
    t_shirt_contour = []
    for point in tshirt_prediction.points:
        t_shirt_contour.append((point.x, point.y))
        corner_coordinates = get_corner_coordinates_for_contour(t_shirt_contour)
        tshirt_prediction.corner_coordinate = corner_coordinates
    return predictions

def correct_class_for_sleeves(predictions):
    # Deep copy
    #print(predictions)
    sleeve_predictions = [x for x in predictions if "sleeve" in x.class_]
    tshirt_prediction = [x for x in predictions if "t_shirt" in x.class_][0]
    #print("top cord", tshirt_prediction.corner_coordinate.top_coordinate)
    #print(sleeve_predictions)
    remove_predictions = []

    WIDTH = []
    for index, prediction in enumerate(predictions):
        contour = []
        feet_prediction = False

        for point in prediction.points:
            if point.y > tshirt_prediction.corner_coordinate.bottom_coordinate[1]:
                #print("think there is")
                remove_predictions.append(prediction)
                feet_prediction = True
                break

            contour.append((point.x, point.y))
        if feet_prediction:
            #print("found one")
            #feet_coordinates = get_corner_coordinates_for_contour(contour)
            #width = (feet_coordinates.right_coordinate[1] - feet_coordinates.left_coordinate[1])
            #print(width)
            #WIDTH.append(width)
            continue
        sleeve_coordinates = get_corner_coordinates_for_contour(contour)
        #print("Corner Coordinates")
        #print(prediction)
        #print(sleeve_coordinates)
        prediction.corner_coordinate = sleeve_coordinates
        width = (sleeve_coordinates.bottom_coordinate[1] - sleeve_coordinates.top_coordinate[1])
        height = (sleeve_coordinates.right_coordinate[0] - sleeve_coordinates.left_coordinate[0])
        #print("Heyaaa")
        #print(width/height)
        #if width/height < 1.5 and width/height > 0.7:
         #   continue
        #else:
          #  remove_predictions.append(prediction)
    sleeve_predictions_list = list(sleeve_predictions)
    #print("sleeve_predictions_list", sleeve_predictions_list)
    for prediction in remove_predictions:
        sleeve_predictions_list.remove(prediction)
    #print("sleeve_predictions_list", sleeve_predictions_list)
    left_sleeve = min(sleeve_predictions_list, key = lambda z : z.x)
    left_sleeve.class_ = "left_sleeve_verified"
    right_sleeve = max(sleeve_predictions_list, key=lambda z: z.x + z.width)
    right_sleeve.class_ = "right_sleeve_verified"
    #feet_width = sum(WIDTH)/len(WIDTH)
    #print("feet_width", feet_width)

    return predictions








def get_corner_coordinates_for_contour(contour):
    corner_coordinates = CornerCoordinate()
    left_index = get_index_of_left_corner_pt(contour)
    corner_coordinates.left_coordinate = contour[left_index]
    right_index = get_index_of_right_corner_pt(contour)
    corner_coordinates.right_coordinate = contour[right_index]
    top_index = get_index_of_top_corner_pt(contour)
    corner_coordinates.top_coordinate = contour[top_index]
    bottom_index = get_index_of_bottom_corner_pt(contour)
    corner_coordinates.bottom_coordinate = contour[bottom_index]
    return corner_coordinates


def get_gradient_quadrant_for_contour(contour):
    x = []
    y = []
    #print(contour)
    for point in contour:
        x.append(point[0])
        y.append(point[1])

    plt.scatter(x, y)




    gradient_quadrant = []
    left_index = get_index_of_left_corner_pt(contour)
    gradient_quadrant.append(get_gradient_at_point(contour, left_index))


    right_index = get_index_of_right_corner_pt(contour)
    gradient_quadrant.append(get_gradient_at_point(contour, right_index))



    top_index = get_index_of_top_corner_pt(contour)
    gradient_quadrant.append(get_gradient_at_point(contour, top_index))
    #cv2.circle(marked_image, tuple(int(cord) for cord in contour[top_index]), 4, (255, 255, 0), 3)


    bottom_index = get_index_of_bottom_corner_pt(contour)
    gradient_quadrant.append(get_gradient_at_point(contour, bottom_index))
    print(contour[bottom_index])

    #cv2.circle(marked_image, tuple(int(cord) for cord in contour[bottom_index]), 4, (255, 255, 0), 3)

    print(gradient_quadrant)
    #cv2.imwrite("sleeves_corners.jpg",marked_image )
    # Show the plot

    plt.savefig('sleeves.png')


    return gradient_quadrant

