import cv2
import os
import numpy as np
from skimage.filters import threshold_sauvola

input_folder = "images"
output_folder = "preprocessed_images"


for filename in os.listdir(input_folder):

    if filename.endswith(".jpg"):

        input_path = os.path.join(input_folder,filename)

        image = cv2.imread(input_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blurred = cv2.GaussianBlur(gray, (3,3),0)


        window_size = 61

        thresh_sauvola = threshold_sauvola(gray, window_size=window_size, k=0.13)

        binary = (gray > thresh_sauvola).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5,5)
        )


        # cleaned = cv2.morphologyEx(
        #     binary,
        #     cv2.MORPH_CLOSE,
        #     kernel,
        # )



        output_path = os.path.join(output_folder, filename)

        cv2.imwrite(output_path,binary)


print("Finished")

