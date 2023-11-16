
from collections import namedtuple
from decimal import Decimal
import math
import numpy as np
import cv2


CornerPoint = namedtuple\
    ('CornerPoint', ['coordinates', 'border_contour_index'])

class TShirtBuilder:

    def __init__(self):

        self.LEFT_CHEST_CORNER_PT = CornerPoint(None, None)
        self.LEFT_WAIST_CORNER_PT = CornerPoint(None, None)
        self.RIGHT_WAIST_CORNER_PT = CornerPoint(None, None)
        self.RIGHT_CHEST_CORNER_PT = CornerPoint(None, None)
        self.LEFT_SHOULDER_PT = CornerPoint(None, None)
        self.RIGHT_SHOULDER_PT = CornerPoint(None, None)

        self.body_length = Decimal()
        self.chest_length = Decimal()
        self.shoulder_length = Decimal()


    def build_body_length(self):
        self.body_length = self.RIGHT_WAIST_CORNER_PT.coordinates[1] \
                           - self.RIGHT_SHOULDER_PT.coordinates[1]
        return self

    def build_chest_length(self):
        self.chest_length = self.RIGHT_CHEST_CORNER_PT.coordinates[0] - self.LEFT_CHEST_CORNER_PT.coordinates[0]
        return self

    def build_shoulder_length(self):
        self.shoulder_length = self.RIGHT_SHOULDER_PT.coordinates[0] - self.LEFT_SHOULDER_PT.coordinates[0]
        return self



    def build_arm_hole(self):
        x1 = self.RIGHT_SHOULDER_PT.coordinates[0]
        x2 = self.RIGHT_CHEST_CORNER_PT.coordinates[0]
        y1 = self.RIGHT_SHOULDER_PT.coordinates[1]
        y2 = self.RIGHT_CHEST_CORNER_PT.coordinates[1]
        self.arm_hole = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))
        return self


    def get_key_points_with_marking(self, image, border_contour):

        x1, y1 = self.LEFT_WAIST_CORNER_PT.coordinates[0], self.LEFT_WAIST_CORNER_PT.coordinates[1]
        x2, y2 = self.RIGHT_WAIST_CORNER_PT.coordinates[0], self.RIGHT_WAIST_CORNER_PT.coordinates[1]

        line_color = (0, 0, 255)
        line_thickness = 1
        cv2.line(image, (x1, y1), (x2, y2), (255,0,0), 3)
        coords = border_contour[self.LEFT_WAIST_CORNER_PT.border_contour_index
                                     :self.RIGHT_WAIST_CORNER_PT.border_contour_index]
        coords = coords.reshape(coords.shape[0],2).tolist()
        print("coordinates of baseline", coords[0], coords[len(coords)-1])
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        # Perform linear regression to find the best-fit line
        coefficients = np.polyfit(x_coords, y_coords, 1)
        m, b = coefficients  # m is the slope, b is the y-intercept
        print("Slope of the base line", m,b)

        # Given coordinates of the point
        x0 = self.LEFT_COLLAR_PT.coordinates[0]
        y0 = self.LEFT_COLLAR_PT.coordinates[1]

        # Calculate the perpendicular distance
        d = abs(y0 - (m * x0 + b)) / math.sqrt(1 + m ** 2)

        #print("Perpendicular distance from the line:", d)

        # Calculate the slope of the perpendicular line
        m_perpendicular = -1 / m

        # Calculate the y-intercept of the perpendicular line
        c_perpendicular = y0 - m_perpendicular * x0

        # Calculate the x-coordinate of the intersection point
        x_intersect= int((c_perpendicular - b) / (m - m_perpendicular))

        # Calculate the y-coordinate of the intersection point
        y_intersect= int(m * x_intersect + b)
        #print("intersects", x_intersect,y_intersect)
        cv2.line(image, (x0, y0), (x_intersect, y_intersect), (255,0,0), 3)
        cv2.imwrite("image_with_key_points.jpg", image)

        #print(coords)
        #cv2.imshow("Marked Points", self.read_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



