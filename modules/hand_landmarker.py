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

import plotly.graph_objects as go


class Results(NamedTuple):
    multi_hand_landmarks: List[NormalizedLandmarkList]
    multi_hand_world_landmarks: List[LandmarkList]
    multi_handedness: List[ClassificationList]


class GestureData(TypedDict):
    GestureName: str
    Images: List[np.ndarray]
    LandmarksData: List[np.ndarray]
    Frames: int
    Graph: go.Figure


class HandLandmarker:
    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2) -> None:
        self.hand_landmarker = hands.Hands(static_image_mode, max_num_hands)

    def process(self, image: ndarray) -> Results:
        return self.hand_landmarker.process(image)

    def draw_landmarks(
        self,
        image: ndarray,
        results: Results,
        landmark_radius: int = 2,
        landmark_thickness: int = 3,
        connection_thickness: int = 3,
        connection_color: Tuple[int, int, int] = (255, 255, 255),
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

    def register_hand_gesture(
        self,
        name: str,
        frames: int = 150,
        font_scale: int = 2,
        wait_frames: int = 150,
        font_thickness: int = 3,
        font_style: int = opencv.FONT_HERSHEY_DUPLEX,
        font_color: Tuple[int, int, int] = (255, 255, 0),
    ) -> bool:
        video = opencv.VideoCapture(0)

        landmark_results: List[List[LandmarkList]] = []
        results_list: List[Results] = []
        starting_frames = frames

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

            results: Results = self.process(
                opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
            )

            results_list.append(results)

            frames -= 1
            if frames == 0:
                break

            if not results.multi_hand_landmarks:
                opencv.imshow("Hand Landmarks", image)
                opencv.waitKey(1)

                continue

            landmark_results.append(results.multi_hand_world_landmarks)
            self.draw_landmarks(image, results)

            opencv.imshow("Hand Landmarks", image)
            opencv.waitKey(1)

        opencv.destroyAllWindows()
        video.release()

        final_landmarks: List[np.ndarray] = [
            np.concatenate(
                [
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in result.landmark
                        ]
                    )
                    for result in results
                ]
            )
            for results in landmark_results
        ]

        images: List[np.ndarray] = []

        steps = 1
        if len(results_list) >= 10:
            steps = len(results_list) // 10

        for i in range(0, len(results_list), steps):
            result = results_list[i]
            image = np.zeros((480, 640, 3), np.uint8)
            image[:, :] = (11051 // 256, 11051 // 256, 11051 // 256)

            if result.multi_hand_landmarks != None:
                self.draw_landmarks(image, result, landmark_color=(0, 255, 255))
                images.append(image)

        fig = go.Figure()
        for index, sub_array in enumerate(final_landmarks):
            x_values = [item[0] for item in sub_array]
            y_values = [item[1] for item in sub_array]
            z_values = [item[2] for item in sub_array]

            fig.add_trace(
                go.Scatter3d(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=index,
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                    name=f"Frame {index+1}",
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis",
            )
        )

        gesture_data = GestureData(
            GestureName=name,
            LandmarksData=final_landmarks,
            Images=images,
            Frames=starting_frames,
            Graph=fig,
        )

        with open(f"landmarks/{name}.pkl", "wb") as file:
            pickle.dump(gesture_data, file)
