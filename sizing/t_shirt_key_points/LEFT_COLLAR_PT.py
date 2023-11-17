import numpy as np
import matplotlib.pyplot as plt
def LEFT_COLLAR_PT(predictions, t_shirt_builder, mIndex):
    t_shirt_contour = []
    t_shirt = [x for x in predictions if x.class_ == "t_shirt"][0]
    for point in t_shirt.points:
        t_shirt_contour.append((point.x, point.y))
    t_shirt_contour = np.array(t_shirt_contour)


    for fIndex, contour in enumerate(t_shirt_contour):
        if fIndex > mIndex:
            slope = abs((t_shirt_contour[fIndex + 5][1] - t_shirt_contour[fIndex][1]) / (
                    t_shirt_contour[fIndex + 5][0] - t_shirt_contour[fIndex][0]))

            if t_shirt_contour[fIndex + 1][1] <= t_shirt_contour[fIndex][1] and  slope < 1.8:
                #print(slope)
                continue
            else:
                print("Found Left Collar")
                print(contour[0], contour[1])
                #print(slope)
                plt.scatter(int(t_shirt_contour[fIndex][0]), int(t_shirt_contour[fIndex][1]), c='black',
                         marker='o', s=100, label='Changed Point')
                t_shirt_builder.LEFT_COLLAR_PT = t_shirt_builder.LEFT_COLLAR_PT._replace(
                    coordinates=(int(t_shirt_contour[fIndex][0]), int(t_shirt_contour[fIndex][1])),
                    border_contour_index=fIndex - 1
                )
                break
    plt.savefig('sizing\scatter_plot.png')
