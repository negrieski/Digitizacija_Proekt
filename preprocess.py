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

        blurred = cv2.GaussianBlur(gray, (3,3),0)

        # 2. Sauvola thresholding (интелигентна бинаризација за OCR)
        window_size = 25
        thresh_sauvola = threshold_sauvola(blurred, window_size=window_size, k=0.2)
        binary = (blurred > thresh_sauvola).astype(np.uint8) * 255
        inverted = cv2.bitwise_not(binary)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted)

        cleaned = np.zeros_like(inverted)


        for i in range(1,num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if 5 <= area <= 4000 and w<250 and h<250:
                cleaned [ labels == i] = 255




        final_binary = cv2.bitwise_not(cleaned)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, final_binary)

print("Finished")

