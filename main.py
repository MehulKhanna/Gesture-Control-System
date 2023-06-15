import cv2 as opencv
from typing import NamedTuple, List
from mediapipe.python.solutions import hands, drawing_utils
from mediapipe.framework.formats.classification_pb2 import ClassificationList
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmarkList,
    LandmarkList,
)


class Results(NamedTuple):
    multi_hand_landmarks: List[NormalizedLandmarkList]
    multi_hand_world_landmarks: List[LandmarkList]
    multi_handedness: List[ClassificationList]


hand_landmarker = hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)

video = opencv.VideoCapture(0)
while video.isOpened():
    success, image = video.read()
    if not success:
        break

    rgb_image = opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
    results: Results = hand_landmarker.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(image, hand_landmarks, hands.HAND_CONNECTIONS)

    opencv.imshow("Hand Landmarks", image)
    if opencv.waitKey(5) & 0xFF == ord("q"):
        break

video.release()
opencv.destroyAllWindows()
