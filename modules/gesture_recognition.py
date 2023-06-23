import os
import pickle
import numpy as np
import cv2 as opencv
from typing import Dict, List, Tuple
from modules.hand_landmarker import HandLandmarker, Results, GestureData


class GestureRecognition:
    def __init__(
        self,
        hand_landmarker: HandLandmarker,
        font_scale: int = 2,
        display: bool = True,
        font_thickness: int = 3,
        font_style: int = opencv.FONT_HERSHEY_DUPLEX,
        font_color: Tuple[int, int, int] = (255, 255, 0),
    ) -> None:
        self.display = display
        self.font_scale = font_scale
        self.font_style = font_style
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.gestures: List[GestureData] = []
        self.hand_landmarker = hand_landmarker

        for root, _, files in os.walk(r"landmarks"):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, "rb") as file:
                    self.gestures.append(pickle.load(file))

    async def start(self) -> None:
        video = opencv.VideoCapture(0)

        while video.isOpened():
            success, image = video.read()
            if not success:
                break

            results: Results = await self.hand_landmarker.process(
                opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
            )

            if not results.multi_hand_landmarks:
                if self.display:
                    opencv.imshow("Gesture Recognition", image)
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

            landmarks: np.ndarray = np.concatenate(landmarks)
            minimum_euclidean_distances: Dict[GestureData, float] = {}

            for gesture in self.gestures:
                calculated_euclidean_distances = [
                    np.linalg.norm(hand_landmarks - landmarks)
                    for hand_landmarks in gesture["Landmarks"]
                    if len(landmarks) == len(hand_landmarks)
                ]

                if calculated_euclidean_distances != []:
                    minimum_euclidean_distances[gesture["Name"]] = min(
                        calculated_euclidean_distances
                    )

            if minimum_euclidean_distances == {}:
                continue

            text = "Nothing"
            pair = min(minimum_euclidean_distances.items(), key=lambda x: x[1])
            if pair[1] < 0.075:
                text = pair[0]

            if self.display:
                await self.hand_landmarker.draw_landmarks(image, results)

                image = opencv.putText(
                    img=image,
                    color=self.font_color,
                    fontFace=self.font_style,
                    org=(160, 90),
                    fontScale=self.font_scale,
                    thickness=self.font_thickness,
                    text=text,
                )

                opencv.imshow("Gesture Recognition", image)
                opencv.waitKey(1)

                continue

        video.release()
        opencv.destroyAllWindows()
