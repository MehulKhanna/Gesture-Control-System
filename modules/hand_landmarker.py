import math
import pickle

import numpy as np
import cv2 as opencv
from numpy import ndarray
from typing import List, NamedTuple, Tuple, TypedDict

from mediapipe.framework.formats.classification_pb2 import ClassificationList
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmarkList,
    LandmarkList,
)

from mediapipe.python.solutions import hands, drawing_utils
from mediapipe.python.solutions.drawing_styles import DrawingSpec


class Results(NamedTuple):
    multi_hand_landmarks: List[NormalizedLandmarkList]
    multi_hand_world_landmarks: List[LandmarkList]
    multi_handedness: List[ClassificationList]


class GestureData(TypedDict):
    Name: str
    Landmarks: List[np.ndarray]


class HandLandmarker:
    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2) -> None:
        self.hand_landmarker = hands.Hands(static_image_mode, max_num_hands)

    async def process(self, image: ndarray) -> Results:
        return self.hand_landmarker.process(image)

    async def draw_landmarks(
        self,
        image: ndarray,
        results: Results,
        landmark_radius: int = 2,
        landmark_thickness: int = 3,
        connection_thickness: int = 3,
        connection_color: Tuple[int, int, int] = (0, 0, 0),
        landmark_color: Tuple[int, int, int] = (255, 255, 0),
    ) -> None:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                image,
                hand_landmarks,
                hands.HAND_CONNECTIONS,
                DrawingSpec(landmark_color, landmark_thickness, landmark_radius),
                DrawingSpec(connection_color, connection_thickness),
            )

    async def register_hand_gesture(
        self,
        name: str,
        frames: int = 150,
        font_scale: int = 2,
        wait_frames: int = 90,
        font_thickness: int = 3,
        font_style: int = opencv.FONT_HERSHEY_DUPLEX,
        font_color: Tuple[int, int, int] = (255, 255, 0),
    ) -> bool:
        landmarks_list: List[np.ndarray] = []

        video = opencv.VideoCapture(0)
        while video.isOpened():
            success, image = video.read()
            if not success:
                break

            if wait_frames <= 0:
                text = f"{math.ceil(frames / 30)}"

            else:
                text = f"{math.ceil(wait_frames / 30)}"

            image_height, image_width = image.shape[:2]

            (text_width, text_height), _ = opencv.getTextSize(
                text, font_style, font_scale, font_thickness
            )

            text_x = (image_width - text_width) // 2
            text_y = (image_height + text_height) // 2

            image = opencv.putText(
                img=image,
                color=font_color,
                fontFace=font_style,
                org=(text_x, text_y),
                fontScale=font_scale,
                thickness=font_thickness,
                text=text,
            )

            wait_frames -= 1
            if wait_frames > 0:
                opencv.imshow("Hand Landmarks", image)
                opencv.waitKey(1)

                continue

            results: Results = await self.process(
                opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
            )

            frames -= 1
            if frames == 0:
                break

            if not results.multi_hand_landmarks:
                opencv.imshow("Hand Landmarks", image)
                opencv.waitKey(1)

                continue

            landmarks: List[np.ndarray] = []
            for hand_landmarks in results.multi_hand_world_landmarks:
                landmarks.append(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in hand_landmarks.landmark
                        ]
                    )
                )

            landmarks_list.append(np.concatenate(landmarks))
            await self.draw_landmarks(image, results)

            opencv.imshow("Hand Landmarks", image)
            opencv.waitKey(1)

        video.release()
        opencv.destroyAllWindows()

        final_data: GestureData = {
            "Name": name,
            "Landmarks": landmarks_list,
        }

        with open(f"landmarks/{name}.pkl", "wb") as file:
            pickle.dump(final_data, file)

        return final_data
