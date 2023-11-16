import cv2
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt


def RIGHT_WAIST_CORNER_PT(predictions, t_shirt_builder, mIndex):
    t_shirt = [x for x in predictions if x.class_ == "t_shirt"][0]
    t_shirt_contour = []
    for point in t_shirt.points:
        t_shirt_contour.append((point.x, point.y))
    t_shirt_contour = np.array(t_shirt_contour)
    break_both_loops = False
    fIndex = 0
    while True:

        if break_both_loops:
            break
        if fIndex > mIndex:


            #print("Hey1")
            #print(fIndex)
            #print(len(t_shirt_contour) - 1)
            #print(str(t_shirt_contour[fIndex][0]) + "-" + str(t_shirt_contour[fIndex][1]))
            #print((abs(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][1] - t_shirt_contour[(fIndex + 10)%(len(t_shirt_contour)-1)][1]) / t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][
            #    1]))
            #print(abs(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][0] -
             #           t_shirt_contour[(fIndex + 10)%(len(t_shirt_contour)-1)][0])
              #      /t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][0])


            if ((abs(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][1]
                     - t_shirt_contour[(fIndex + 10)%(len(t_shirt_contour)-1)][1])
                 / t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][
                1]) < 0.04) and (
                    abs(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][0] -
                        t_shirt_contour[(fIndex + 10)%(len(t_shirt_contour)-1)][0])
                    /t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][0] > 0.01):

                while True:
                    if t_shirt_contour[(fIndex + 1)%(len(t_shirt_contour)-1)][1] > t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][1]:
                        fIndex = fIndex + 1
                    else:

                        print("found Right Waist Corner")
                        print(int(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][0]), int(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][1]))
                        plt.scatter(int(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][0]), int(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][1]), c='black',
                                            marker='o', s=100, label='Changed Point')

                        t_shirt_builder.RIGHT_WAIST_CORNER_PT = t_shirt_builder.RIGHT_WAIST_CORNER_PT._replace(
                                    coordinates=(int(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][0]), int(t_shirt_contour[fIndex%(len(t_shirt_contour)-1)][1])),
                                    border_contour_index=fIndex)
                        plt.savefig('sleeves_1.png')

                        break_both_loops = True
                        break



            else:
                fIndex = fIndex + 1


        else:
            fIndex = fIndex + 1

