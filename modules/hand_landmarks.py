import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple


def draw_landmarks(
    image: mp.Image,
    hand_landmarker_result: mp.tasks.vision.HandLandmarkerResult,
    connection_color: Tuple[int, int, int] = (255, 255, 255),
    connection_thickness: int = 2,
    landmark_radius: int = 6,
    landmark_color: Tuple[int, int, int] = (0, 0, 0),
    landmark_thickness: int = 2,
    bounding_box_padding: int = 20,
) -> np.ndarray:
    image = image.numpy_view()
    for landmarks in hand_landmarker_result.hand_landmarks:
        height, width, _ = image.shape

        x_coordinates = [int(landmark.x * width) for landmark in landmarks]
        y_coordinates = [int(landmark.y * height) for landmark in landmarks]
        coordinates = [
            (x, y_coordinates[index]) for index, x in enumerate(x_coordinates)
        ]

        [
            cv2.line(
                img=image,
                pt1=coordinates[connection.start],
                pt2=coordinates[connection.end],
                color=connection_color,
                thickness=connection_thickness,
            )
            for connection in mp.tasks.vision.HandLandmarksConnections().HAND_CONNECTIONS
        ]

        [
            cv2.circle(
                img=image,
                center=coordinate,
                radius=landmark_radius,
                color=landmark_color,
                thickness=landmark_thickness,
            )
            for coordinate in coordinates
        ]

        xmin, ymin = min(x_coordinates), min(y_coordinates)
        xmax, ymax = max(x_coordinates), max(y_coordinates)

    return image[
        -bounding_box_padding + ymin : ymax + bounding_box_padding,
        -bounding_box_padding + xmin : xmax + bounding_box_padding,
    ]


def detect_hand_landmarks(
    image: mp.Image,
    model_path: str = r"models/hand_landmarker.task",
    number_of_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.tasks.vision.HandLandmarkerResult:
    hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model_path,
        ),
        num_hands=number_of_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        hand_landmarker_options
    )

    hand_landmarker_results = hand_landmarker.detect(image)
    return hand_landmarker_results
