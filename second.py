import cv2
import numpy as np
import pytesseract
import pandas as pd

img = cv2.imread("Image3.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours top-to-bottom, then left-to-right
contours = sorted(
    contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0])
)

# Define empty list to store cell values
cells = []

# Define padding
padding = 10

# Iterate through each contour
for cnt in contours:
    # Get bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # Make sure contour area is big enough
    if cv2.contourArea(cnt) > 10:
        # Add padding to bounding box
        roi = gray[
            max(0, y - padding) : min(img.shape[0], y + h + padding),
            max(0, x - padding) : min(img.shape[1], x + w + padding),
        ]

        # Extract the cell text
        cell_text = pytesseract.image_to_string(roi)

        # Clean the cell text
        cell_text = " ".join(cell_text.split())
        cell_text = cell_text.replace("\n", " ")
        cell_text = cell_text.replace("\x0c", " ")

        # If cell is empty, assign 'empty'
        if not cell_text:
            cell_text = "empty"

        # Append cell text to cells list
        cells.append((x, y, cell_text))

# Sort the cells by their y-coordinate and then by their x-coordinate
cells.sort(key=lambda tup: (tup[1], tup[0]))

# Calculate the number of columns
num_cols = next(
    (i for i in range(1, len(cells)) if abs(cells[i][1] - cells[0][1]) > h), len(cells)
)

# Convert the cells list into a grid structure
grid = [cells[i : i + num_cols] for i in range(0, len(cells), num_cols)]

# Convert the grid to a DataFrame, keeping only the cell text and dropping the coordinates
df = pd.DataFrame([[cell[2] for cell in row] for row in grid])

# Print DataFrame
print(df)
df.to_csv("output.csv", index=False, header=False)
