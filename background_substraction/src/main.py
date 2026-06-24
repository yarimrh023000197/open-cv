import numpy as np
import cv2
import os

os.makedirs('out', exist_ok=True)

# Load video file
cap = cv2.VideoCapture('data/traffic_car.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_original = cv2.VideoWriter('out/original.mp4', fourcc, fps, (width, height))
out_mask     = cv2.VideoWriter('out/foreground_mask.mp4', fourcc, fps, (width, height), isColor=False)

# Create background subtractor (MOG2 handles shadows well)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break    # Stop if video ends

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    out_original.write(frame)
    out_mask.write(fgmask)

    # Show original and foreground mask side by side
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # Press 'Esc' to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
out_original.release()
out_mask.release()