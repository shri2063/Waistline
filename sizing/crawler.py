from sizing.t_shirt_builder import TShirtBuilder
from sizing.t_shirt_key_points.LEFT_CHEST_CORNER_PT import LEFT_CHEST_CORNER_PT
from sizing.t_shirt_key_points.LEFT_COLLAR_PT import LEFT_COLLAR_PT
from sizing.t_shirt_key_points.LEFT_SHOULDER_PT import *
import json
from sizing.t_shirt_key_points.LEFT_WAIST_CORNER_PT import LEFT_WAIST_CORNER_PT
from sizing.t_shirt_key_points.RIGHT_CHEST_CORNER_PT import RIGHT_CHEST_CORNER_PT

from sizing.t_shirt_key_points.RIGHT_SHOULDER_PT import RIGHT_SHOULDER_PT
from sizing.t_shirt_key_points.RIGHT_WAIST_CORNER_PT import RIGHT_WAIST_CORNER_PT
MOBILE_LENGTH = 15.8
MOBILE_WIDTH = 7.3


def get_ratios_for_tshirt(predictions):
    chest_length, shoulder_length, tshirt_length = build_t_shirt_key_points(predictions)
    print("C/S: ")
    print(round(chest_length / shoulder_length, 2))
    print("C/L: ")
    print(round(chest_length / tshirt_length, 2))
    print("S/L: ")
    print(round(shoulder_length / tshirt_length, 2))
    return (round(chest_length / shoulder_length, 2),round(chest_length / tshirt_length, 2)
            ,round(shoulder_length / tshirt_length, 2))
def get_actual_length_for_tshirt(predictions):
    chest_length, shoulder_length, tshirt_length = build_t_shirt_key_points(predictions)
    p_x, p_y = get_pixel_count_for_one_cm(predictions)
    print("Chest:", chest_length / p_x * 0.97)
    print("Shoulder:", shoulder_length / (p_x * 0.95))
    print("Tshirt:", tshirt_length / p_y)
    return (chest_length / (p_x * 0.97)
            , shoulder_length / (p_x * 0.95), tshirt_length / p_y)


def get_pixel_count_for_one_cm(predictions):
    mobile_contour = []


    mobile_class = [x for x in predictions if x.class_ == "mobile"][0]
    for point in mobile_class.points:
        mobile_contour.append((point.x, point.y))
    mobile_contour = np.array(mobile_contour)
    # print("mobile")
    x_coordinates = mobile_contour[:, 0]
    y_coordinates = mobile_contour[:, 1]
    # Find min and max values along each axis
    min_x, max_x = np.min(x_coordinates), np.max(x_coordinates)
    min_y, max_y = np.min(y_coordinates), np.max(y_coordinates)
    x_avg = (min_x + max_x) / 2
    y_avg = (min_y + max_y) / 2
    indices = np.where(y_coordinates < y_avg)
    # print("indices", indices)
    filtered_top_coordinates = x_coordinates[indices]
    top_min_x, top_max_x = np.min(filtered_top_coordinates), np.max(filtered_top_coordinates)
    indices = np.where(y_coordinates > y_avg)
    filtered_bottom_coordinates = x_coordinates[indices]
    bot_min_x, bot_max_x = np.min(filtered_bottom_coordinates), np.max(filtered_bottom_coordinates)
    indices = np.where(x_coordinates < x_avg)
    filtered_left_coordinates = y_coordinates[indices]
    left_min_y, left_max_y = np.min(filtered_left_coordinates), np.max(filtered_left_coordinates)
    indices = np.where(x_coordinates > x_avg)
    filtered_right_coordinates = y_coordinates[indices]
    right_min_y, right_max_y = np.min(filtered_right_coordinates), np.max(filtered_right_coordinates)

    print("Top X:", top_min_x, top_max_x)
    print("Bot X:", bot_min_x, bot_max_x)
    print("Left Y:", left_min_y, left_max_y)
    print("Right Y:", right_min_y, right_max_y)
    p_x = (((top_max_x - top_min_x) + (bot_max_x - bot_min_x)) / 2) / MOBILE_WIDTH
    p_y = (((left_max_y - left_min_y) + (right_max_y - right_min_y)) / 2) / MOBILE_LENGTH
    p_y = p_y * 0.95
    if p_x > 1.2 * p_y:
        p_x = 0.95 * p_x
    return p_x, p_y


def build_t_shirt_key_points(predictions):


    t_shirt_builder = TShirtBuilder()
    t_shirt_contour = []
    t_shirt_class = [x for x in predictions if x.class_ == "t_shirt"][0]
    for point in t_shirt_class.points:
        t_shirt_contour.append((point.x, point.y))
    t_shirt_contour = np.array(t_shirt_contour)
    t_shirt_contour_json = json.dumps(t_shirt_contour.tolist())
    #with open("sizing/t_shirt_contour.json", "w+") as json_file:
        #json.dump(t_shirt_contour_json, json_file, indent=2)



    LEFT_WAIST_CORNER_PT(predictions, t_shirt_builder)
    LEFT_CHEST_CORNER_PT(predictions, t_shirt_builder)
    LEFT_SHOULDER_PT(predictions, t_shirt_builder, t_shirt_builder.LEFT_CHEST_CORNER_PT.border_contour_index + 1)
    LEFT_COLLAR_PT(predictions, t_shirt_builder, t_shirt_builder.LEFT_SHOULDER_PT.border_contour_index + 1)
    RIGHT_SHOULDER_PT(predictions, t_shirt_builder, t_shirt_builder.LEFT_SHOULDER_PT.border_contour_index + 1)
    RIGHT_CHEST_CORNER_PT(predictions,  t_shirt_builder , t_shirt_builder.RIGHT_SHOULDER_PT.border_contour_index + 1)
    RIGHT_WAIST_CORNER_PT(predictions,  t_shirt_builder , t_shirt_builder.RIGHT_CHEST_CORNER_PT.border_contour_index + 1)
    chest_length = (t_shirt_builder.RIGHT_CHEST_CORNER_PT.coordinates[0] - t_shirt_builder.LEFT_CHEST_CORNER_PT.coordinates[0])/1

    print("chest length: " + str(chest_length) )
    shoulder_length = (t_shirt_builder.RIGHT_SHOULDER_PT.coordinates[0] - \
                   t_shirt_builder.LEFT_SHOULDER_PT.coordinates[0])/1
    print("shoulder length: " + str(shoulder_length))
    t_shirt_contour = t_shirt_contour.reshape(t_shirt_contour.shape[0],2).tolist()
    coords_left = t_shirt_contour[0:t_shirt_builder.LEFT_WAIST_CORNER_PT.border_contour_index]
    coords_right = t_shirt_contour[t_shirt_builder.RIGHT_WAIST_CORNER_PT.border_contour_index: len(t_shirt_contour) - 1]
    if t_shirt_builder.LEFT_WAIST_CORNER_PT.border_contour_index < t_shirt_builder.RIGHT_WAIST_CORNER_PT.border_contour_index:
        coords = coords_left + coords_right
    else:
        coords = coords_left


    y_coords = [coord[1] for coord in coords]
    average_y_waist = sum(y_coords) / len(y_coords)
    y_neck = t_shirt_builder.LEFT_COLLAR_PT.coordinates[1]
    plt.scatter(int(t_shirt_class.corner_coordinate.top_coordinate[0]), int(t_shirt_class.corner_coordinate.top_coordinate[1]), c='black',
               marker='o', s=100, label='Changed Point')
    plt.savefig('sizing\scatter_plot.png')

    tshirt_length = (average_y_waist - y_neck)/1
    print("tshirt length: " + str(tshirt_length))
    return chest_length , shoulder_length, tshirt_length


















