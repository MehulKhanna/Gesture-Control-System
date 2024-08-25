import os
import pickle
import numpy as np
import cv2 as opencv
from typing import List, Tuple
from modules.hand_landmarker import HandLandmarker, GestureData


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

    def reload_gestures(self) -> None:
        self.gestures: List[GestureData] = []

        for root, _, files in os.walk(r"landmarks"):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, "rb") as file:
                    self.gestures.append(pickle.load(file))

    def start(self) -> None:
        video = opencv.VideoCapture(0)

        while video.isOpened():
            success, image = video.read()
            if not success:
                break

            results = self.hand_landmarker.process(
                opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
            )

            if not results.multi_hand_landmarks:
                opencv.imshow("Gesture Recognition", image)
                if opencv.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            final_landmarks: np.ndarray = np.concatenate(
                [
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in hand_landmarks.landmark
                        ]
                    )
                    for hand_landmarks in results.multi_hand_world_landmarks
                ]
            )

            euclidean_distances: List[(GestureData, List[float])] = [
                (
                    gesture,
                    [
                        np.linalg.norm(landmarks - final_landmarks)
                        for landmarks in gesture["LandmarksData"]
                        if len(landmarks) == len(final_landmarks)
                    ],
                )
                for gesture in self.gestures
            ]

            if not euclidean_distances:
                self.hand_landmarker.draw_landmarks(image, results)
                opencv.imshow("Gesture Recognition", image)
                if opencv.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            filtered = list(filter(lambda x: x[1], euclidean_distances))
            if not filtered:
                self.hand_landmarker.draw_landmarks(image, results)
                opencv.imshow("Gesture Recognition", image)
                if opencv.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            minimum_euclidean_distance: Tuple[GestureData, float] = min(
                filtered,
                key=lambda x: min(x[1]),
            )

            text = minimum_euclidean_distance[0]["GestureName"]
            if min(minimum_euclidean_distance[1]) < 0.065:
                image = opencv.putText(
                    img=image,
                    color=self.font_color,
                    fontFace=self.font_style,
                    org=(160, 90),
                    fontScale=self.font_scale,
                    thickness=self.font_thickness,
                    text=text,
                )

            self.hand_landmarker.draw_landmarks(image, results)
            opencv.imshow("Gesture Recognition", image)
            if opencv.waitKey(1) & 0xFF == ord("q"):
                break

        video.release()
        opencv.destroyAllWindows()
