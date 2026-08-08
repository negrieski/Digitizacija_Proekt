import cv2
import os

input_folder = "images"
output_folder = "preprocessed_images"


for filename in os.listdir(input_folder):

    if filename.endswith(".jpg"):

        input_path = os.path.join(input_folder,filename)

        image = cv2.imread(input_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        output_path = os.path.join(output_folder, filename)

        cv2.imwrite(output_path,gray)

print("Finished")