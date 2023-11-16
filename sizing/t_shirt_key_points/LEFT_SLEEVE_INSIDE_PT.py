import cv2
import numpy as np


def LEFT_SLEEVE_INSIDE_PT(border_contour, read_image, t_shirt_builder, index):
    reshaped_contour = border_contour.reshape(border_contour.shape[0], 2)

    while True:
        if reshaped_contour[index][1] > reshaped_contour[index + 1][1]:
            coordinates = tuple(reshaped_contour[index])
            t_shirt_builder.LEFT_SLEEVE_INSIDE_PT = t_shirt_builder.LEFT_SHOULDER_PT._replace(
                coordinates=coordinates, border_contour_index=index
            )
            break

        else:
            index = index + 1



    # print("left sleeve outside point", self.left_sleeve_outside_pt)
    cv2.circle(read_image, tuple(coordinates), 25, (255, 0, 0), 3)
    # cv2.imshow("Marked Points", self.read_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
