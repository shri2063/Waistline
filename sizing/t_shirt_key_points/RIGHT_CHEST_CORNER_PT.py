import cv2
import numpy as np
from decimal  import  Decimal

from matplotlib import pyplot as plt


def RIGHT_CHEST_CORNER_PT(predictions, t_shirt_builder,  mIndex):
    right_sleeve = [x for x in predictions if x.class_ == "right_sleeve_verified"][0]
    right_sleeve_contour = []
    for point in right_sleeve.points:
        right_sleeve_contour.append((point.x, point.y))
    right_sleeve_contour = np.array(right_sleeve_contour)
    t_shirt_contour = []
    t_shirt = [x for x in predictions if x.class_ == "t_shirt"][0]
    for point in t_shirt.points:
        t_shirt_contour.append((point.x, point.y))
    t_shirt_contour = np.array(t_shirt_contour)
    break_both_loops = False

    for fIndex, contour in enumerate(t_shirt_contour):
        if fIndex > mIndex:
            sIndex = 0
            while True:
                if (abs(right_sleeve_contour[sIndex][0] - contour[0]) +
                        abs(right_sleeve_contour[sIndex][1] - contour[1]) < 10.0):
                    break

                else:
                    if sIndex == len(right_sleeve_contour) - 1:
                        print("Found Right Chest Corner")
                        print(contour[0], contour[1])
                        plt.scatter(int(t_shirt_contour[fIndex][0]), int(t_shirt_contour[fIndex][1]), c='black',
                                    marker='o', s=100, label='Changed Point')
                        t_shirt_builder.RIGHT_CHEST_CORNER_PT = t_shirt_builder.RIGHT_CHEST_CORNER_PT._replace(
                            coordinates=(int(t_shirt_contour[fIndex][0]), int(t_shirt_contour[fIndex][1])),
                            border_contour_index=fIndex
                        )
                        break_both_loops = True
                        break


                    sIndex = (sIndex + 1)

        if break_both_loops:
            break

    plt.savefig('sleeves_1.png')




