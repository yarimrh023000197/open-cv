import cv2
import numpy as np
import os
from datetime import datetime

# Crear carpetas si no existen
os.makedirs('data', exist_ok=True)
os.makedirs('out', exist_ok=True)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Read the first frame and convert to float
_, img = cap.read()
averageValue1 = np.float32(img)

# Guardamos las últimas frames para capturar al cerrar
last_img = img.copy()
last_background = None

while True:
    # Capture next frame
    _, img = cap.read()
    
    # Update background model
    cv2.accumulateWeighted(img, averageValue1, 0.02)
    
    # Convert back to 8-bit for display
    resultingFrames1 = cv2.convertScaleAbs(averageValue1)

    # Actualizar últimas frames
    last_img = img.copy()
    last_background = resultingFrames1.copy()

    # Show both original and background model
    cv2.imshow('Original Frame', img)
    cv2.imshow('Background (Running Average)', resultingFrames1)
    
    # Exit on Esc key
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Guardar imágenes al cerrar
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

data_path = os.path.join('data', f'original_{timestamp}.png')
out_path  = os.path.join('out',  f'background_{timestamp}.png')

cv2.imwrite(data_path, last_img)
cv2.imwrite(out_path, last_background)

print(f'Imagen original guardada en:    {data_path}')
print(f'Imagen con background guardada: {out_path}')

# Cleanup
cap.release()
cv2.destroyAllWindows()