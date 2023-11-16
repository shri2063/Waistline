from sizing.t_shirt_builder import TShirtBuilder
from sizing.t_shirt_key_points.LEFT_CHEST_CORNER_PT import LEFT_CHEST_CORNER_PT
from sizing.t_shirt_key_points.LEFT_SHOULDER_PT import *
import json
from sizing.t_shirt_key_points.LEFT_WAIST_CORNER_PT import LEFT_WAIST_CORNER_PT
from sizing.t_shirt_key_points.RIGHT_CHEST_CORNER_PT import RIGHT_CHEST_CORNER_PT

from sizing.t_shirt_key_points.RIGHT_SHOULDER_PT import RIGHT_SHOULDER_PT
from sizing.t_shirt_key_points.RIGHT_WAIST_CORNER_PT import RIGHT_WAIST_CORNER_PT
MOBILE_LENGTH = 15.8
MOBILE_WIDTH = 7.3


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

    left_sleeve_contour = []
    left_sleeve_class = [x for x in predictions if x.class_ == "left_sleeve_verified"][0]
    for point in left_sleeve_class.points:
        left_sleeve_contour.append((point.x, point.y))
    left_sleeve_contour = np.array(left_sleeve_contour)
    left_sleeve_contour = json.dumps(left_sleeve_contour.tolist())
    #with open("sizing/left_sleeve_contour.json", "w+") as json_file:
        #json.dump(left_sleeve_contour, json_file, indent=2)

    mobile_contour = []
    p_x = 0
    p_y = 0
    try:
        mobile_class = [x for x in predictions if x.class_ == "mobile"][0]
        for point in mobile_class.points:
            mobile_contour.append((point.x, point.y))
        mobile_contour = np.array(mobile_contour)
        #print("mobile")
        x_coordinates = mobile_contour[:,0]
        y_coordinates = mobile_contour[:, 1]
        # Find min and max values along each axis
        min_x, max_x = np.min(x_coordinates), np.max(x_coordinates)
        min_y, max_y = np.min(y_coordinates), np.max(y_coordinates)
        x_avg = (min_x + max_x)/2
        y_avg = (min_y + max_y) / 2
        indices = np.where(y_coordinates < y_avg)
        #print("indices", indices)
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
        print("Left Y:",left_min_y, left_max_y )
        print("Right Y:", right_min_y, right_max_y)
        p_x = (((top_max_x - top_min_x) + (bot_max_x - bot_min_x))/2)/MOBILE_WIDTH
        p_y = (((left_max_y - left_min_y) + (right_max_y - right_min_y)) / 2)/MOBILE_LENGTH
        p_y = p_y*0.95
        if p_x > 1.2*p_y:
            p_x = 0.95*p_x

    except Exception as e:
        print(e.__str__())

    LEFT_WAIST_CORNER_PT(predictions, t_shirt_builder)
    LEFT_CHEST_CORNER_PT(predictions, t_shirt_builder)
    LEFT_SHOULDER_PT(predictions, t_shirt_builder, t_shirt_builder.LEFT_CHEST_CORNER_PT.border_contour_index + 1)
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
    y_neck = t_shirt_class.corner_coordinate.top_coordinate[1]
    plt.scatter(int(t_shirt_class.corner_coordinate.top_coordinate[0]), int(t_shirt_class.corner_coordinate.top_coordinate[1]), c='black',
               marker='o', s=100, label='Changed Point')
    plt.savefig('sizing\scatter_plot.png')

    tshirt_length = (average_y_waist - y_neck)/1
    print("tshirt length: " + str(tshirt_length))
    print("C/S: " )
    print(round(chest_length/shoulder_length, 2))
    print("C/L: ")
    print(round(chest_length / tshirt_length,2))
    print("S/L: ")
    print(round(shoulder_length / tshirt_length,2))
    print("Chest:" , chest_length/p_x*0.97)
    print("Shoulder:", shoulder_length/(p_x*0.95))
    print("Tshirt:", tshirt_length/p_y)
    return (chest_length/(p_x*0.97)
            ,shoulder_length/(p_x*0.95), tshirt_length/p_y)













class Crawler:


    def __init__(self, border_contour, t_shirt_builder, image, border_contour_1):
        self.border_contour = border_contour
        self.t_shirt_builder = t_shirt_builder
        self.read_image = cv2.imread(image)
        self.polygon_cp = border_contour_1

    def set_attribute(self, variable, value):
        setattr(self.t_shirt_builder, variable, value)

    def get_image(self):
        grid_interval = 30
        grid_colour = (0 ,255 ,0)
        # for y in range(0, self.read_image.shape[0], grid_interval):
         #  cv2.line(self.read_image, (0,y), (self.read_image.shape[1],y),grid_colour,1)
        # for x in range(0, self.read_image.shape[1], grid_interval):
         #  cv2.line(self.read_image, (x,0), (x, self.read_image.shape[0]),grid_colour,1)

        # cv2.imwrite("image_with_grid.jpg",self.read_image)

        plt.imshow(self.read_image)
        plt.show()

        # cv2.imshow("Marked Points", self.read_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



    def find_slope(self, coordinates, origin):
        # print(coordinates)
        # print(origin)
        return abs(coordinates[1] - origin[1]) / abs(coordinates[0] - origin[0])


    def calculate_length(self):
        length_between_points = 0
        print("First point on border contour", tuple(self.border_contour[0][0]), self.border_contour.shape)
        outer_point_index = 0
        inner_point_index = 0
        self.get_left_sleeve_outside_pt()
        self.get_all_corner_points_using_polygon_appx()
        print("sleeve inside point", self.left_sleeve_inside_pt)
        print("sleeve outside point", self.left_sleeve_outside_pt)

        for index, point in enumerate(self.border_contour):
            if tuple(self.border_contour[index][0]) == self.left_sleeve_outside_pt:
                outer_point_index = index
                break
        for index, point in enumerate(self.border_contour):
            if tuple(self.border_contour[index][0]) == self.left_sleeve_inside_pt:
                inner_point_index = index
                break

        # print("left sleeve outside point", left_sleeve_outside_point)
        print("sleeve length ", cv2.arcLength(self.border_contour[outer_point_index:inner_point_index], False))
