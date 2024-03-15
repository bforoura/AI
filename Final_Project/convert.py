# at the command prompt type:
# pip install opencv-python


# OpenCV is a library of programming functions for computer vision applications
import cv2
import numpy as np

# Load the image
image = cv2.imread('floorplan1_nolegend.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load image.")
    exit(1)

# Resize the image to 50x50
resized_image = cv2.resize(image, (50, 50))

# Convert the resized image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Threshold the image
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary = cv2.bitwise_not(binary)

# Convert the binary image to a matrix of 0s and 1s
matrix = (binary / 255).astype(int)

# Save the matrix into a text file
np.savetxt('floorplan_matrix.txt', matrix, fmt='%d')

print("Matrix saved to floorplan_matrix.txt")



