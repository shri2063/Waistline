import cv2
import numpy as np
import matplotlib.pyplot as plt

def LEFT_SHOULDER_PT(predictions, t_shirt_builder, mIndex):

    left_sleeve = [x for x in predictions if x.class_ == "left_sleeve_verified"][0]
    left_sleeve_contour = []
    for point in left_sleeve.points:
        left_sleeve_contour.append((point.x, point.y))
    left_sleeve_contour = np.array(left_sleeve_contour)
    #left_shoulder_coordinate = left_sleeve.corner_coordinate.top_coordinate
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
                if (abs(left_sleeve_contour[sIndex][0] - contour[0]) +
                        abs(left_sleeve_contour[sIndex][1] - contour[1]) < 7.0):
                    break
                else:
                    if sIndex ==  len(left_sleeve_contour) - 1:
                        print("Found Left Shoulder")
                        print(contour[0], contour[1])
                        plt.scatter(int(t_shirt_contour[fIndex][0]), int(t_shirt_contour[fIndex][1]), c='black',
                                marker='o', s=100, label='Changed Point')
                        t_shirt_builder.LEFT_SHOULDER_PT = t_shirt_builder.LEFT_SHOULDER_PT._replace(
                                    coordinates=(int(t_shirt_contour[fIndex][0]), int(t_shirt_contour[fIndex][1])),
                                    border_contour_index=fIndex - 1
                                )
                        break_both_loops = True
                        break
                    sIndex = (sIndex + 1)

        if break_both_loops:
            break

    plt.savefig('sleeves_1.png')


