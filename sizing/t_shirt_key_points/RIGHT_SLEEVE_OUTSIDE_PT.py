import cv2
import numpy as np
from decimal  import  Decimal


def RIGHT_SLEEVE_OUTSIDE_PT(border_contour, read_image, t_shirt_builder, index):
    reshaped_contour = border_contour.reshape(border_contour.shape[0], 2)

    while True:
        if border_contour[index%(len(border_contour) - 1)][0] > border_contour[(index + 1)%(len(border_contour) - 1)][0]:
            coordinates = tuple(reshaped_contour[index])
            t_shirt_builder.RIGHT_SLEEVE_OUTSIDE_PT = t_shirt_builder.RIGHT_SLEEVE_OUTSIDE_PT._replace(
                coordinates=coordinates,  border_contour_index=index
            )
            break

        else:
            index = index + 1
    cv2.circle(read_image, tuple(coordinates), 25, (255, 0, 0), 3)
