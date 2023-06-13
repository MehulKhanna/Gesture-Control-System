import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from modules.hand_landmarks import detect_hand_landmarks, draw_landmarks


image = Image.open(r"images/2023-06-13-211745.jpg")
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))

result = detect_hand_landmarks(image)
image = draw_landmarks(image, result)

cv2.imshow("Hand Gesture Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
