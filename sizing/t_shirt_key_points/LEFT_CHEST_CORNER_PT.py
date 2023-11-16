import cv2
import numpy as np
import matplotlib.pyplot as plt
from _decimal import Decimal



def LEFT_CHEST_CORNER_PT(predictions, t_shirt_builder):
    left_sleeve = [x for x in predictions if x.class_ == "left_sleeve_verified"][0]
    left_sleeve_contour = []
    for point in left_sleeve.points:
        left_sleeve_contour.append((point.x, point.y))
    left_sleeve_contour = np.array(left_sleeve_contour)
    t_shirt = [x for x in predictions if x.class_ == "t_shirt"][0]
    t_shirt_contour = []
    for point in t_shirt.points:
        t_shirt_contour.append((point.x, point.y))
    t_shirt_contour = np.array(t_shirt_contour)
    break_both_loops = False

    for fIndex, contour in enumerate(t_shirt_contour):
        sIndex = 0
        while True:
            if (abs(left_sleeve_contour[sIndex][0] - contour[0]) +
                    abs(left_sleeve_contour[sIndex][1] - contour[1]) < 7.0):
                print("Found Left Chest Corner")
                print(contour[0], contour[1])
                plt.scatter(int(t_shirt_contour[fIndex][0]), int(t_shirt_contour[fIndex][1]), c='black',
                            marker='o', s=100, label='Changed Point')
                t_shirt_builder.LEFT_CHEST_CORNER_PT = t_shirt_builder.LEFT_CHEST_CORNER_PT._replace(
                                coordinates=(int(t_shirt_contour[fIndex - 1][0]), int(t_shirt_contour[fIndex - 1][1])),
                                border_contour_index=fIndex - 1
                            )
                break_both_loops = True
                break
            sIndex = (sIndex + 1)
            if sIndex == len(left_sleeve_contour) - 1:
                break
        if break_both_loops:
                break



    # Add labels and a title

    plt.savefig('sizing\scatter_plot.png')
