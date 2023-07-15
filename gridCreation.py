import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Read the image
img = cv2.imread("timage.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Dilate the image to merge word clusters
vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (20, 185)
)  # changing 35 to extend lines.
dilated = cv2.dilate(thresh, vertical_kernel, iterations=2)

# Perform closing operation
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, vertical_kernel)

# Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

padding = 35  # Padding in pixels

for cnt in contours:
    # Filter contours based on area
    if cv2.contourArea(cnt) > 100:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Calculate the x-coordinate of the right border
        right_border_x = min(img.shape[1], x + w + padding)

        # Draw the right border
        cv2.line(img, (right_border_x, y), (right_border_x, y + h), (0, 0, 0), 2)

# Show the image
cv2.imwrite("Image.png", gray)
cv2.imwrite("Image2.png", img)


# Apply a threshold
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Dilate the image to merge word clusters
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (250, 1))
dilated = cv2.dilate(thresh, horizontal_kernel, iterations=2)

# Perform closing operation
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, horizontal_kernel)

# Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

padding = 15  # Padding in pixels

for cnt in contours:
    # Filter contours based on area
    if cv2.contourArea(cnt) > 100:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Calculate the y-coordinate of the bottom border
        bottom_border_y = min(img.shape[0], y + h + padding)

        # Draw the bottom border
        cv2.line(img, (x, bottom_border_y), (x + w, bottom_border_y), (0, 0, 0), 2)


cv2.imwrite("image3.png", img)
