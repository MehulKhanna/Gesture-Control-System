import os
import pickle
import cv2 as opencv
from PIL import Image
import customtkinter as ctk
from modules.gesture_recognition import GestureRecognition
from modules.hand_landmarker import HandLandmarker, GestureData

ctk.set_appearance_mode("dark")
hand_landmarker = HandLandmarker()
gesture_recognition = GestureRecognition(hand_landmarker)


class SideBar(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            label_text="Gestures",
            label_font=("Century", 22),
            label_fg_color="#333333",
            **kwargs,
        )
        self.pack_propagate(False)
        self.columnconfigure(0, weight=1)

        self.gesture_buttons = [
            ctk.CTkButton(
                self,
                text=gesture["GestureName"],
                font=("Century", 14),
                height=40,
                bg_color="transparent",
                fg_color="#333333",
                hover_color="#373737",
                command=lambda gesture=gesture: self.gesture_button(gesture),
            )
            for gesture in gesture_recognition.gestures
        ]

        [
            button.grid(row=index, column=0, sticky="wes", padx=(0, 2), pady=5)
            for index, button in enumerate(self.gesture_buttons)
        ]

    def gesture_button(self, gesture: GestureData) -> None:
        app.main_view = GestureView(app, gesture)
        app.main_view.grid(column=1, row=0, sticky="news", padx=10, pady=10)

    def reload_scrollbar(self) -> None:
        gesture_recognition.reload_gestures()
        [button.destroy() for button in self.gesture_buttons]

        self.gesture_buttons = [
            ctk.CTkButton(
                self,
                text=gesture["GestureName"],
                font=("Century", 14),
                height=40,
                bg_color="transparent",
                fg_color="#333333",
                hover_color="#373737",
                command=lambda gesture=gesture: self.gesture_button(gesture),
            )
            for gesture in gesture_recognition.gestures
        ]

        [
            button.grid(row=index, column=0, sticky="wes", padx=(0, 2), pady=5)
            for index, button in enumerate(self.gesture_buttons)
        ]


class BottomBar(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.pack_propagate(False)
        self.thread = None
        self.register_popup = None
        self.video: opencv.VideoCapture = None

        self.register_button = ctk.CTkButton(
            self,
            text="Register Gesture",
            font=("Century", 18),
            bg_color="transparent",
            fg_color="#333333",
            hover_color="#373737",
            command=self.register_gesture,
            height=40,
        )

        self.start_recognition_button = ctk.CTkButton(
            self,
            text="Start Recognition",
            font=("Century", 18),
            bg_color="transparent",
            fg_color="#333333",
            hover_color="#373737",
            command=self.start_recognition,
            height=40,
        )

        self.copyright_label = ctk.CTkLabel(
            self, text="© Mehul Khanna", font=("Century", 18)
        )

        self.home_button = ctk.CTkButton(
            self,
            text="Home",
            font=("Century", 18),
            bg_color="transparent",
            fg_color="#333333",
            hover_color="#373737",
            command=self.home,
            height=40,
        )

        self.register_button.pack(side="left", padx=(20, 10))
        self.start_recognition_button.pack(side="left", padx=(10, 10))
        self.copyright_label.pack(side="left", padx=(10, 20))
        self.home_button.pack(side="right", padx=(10, 20))

    def home(self) -> None:
        app.main_view = MainView(app)
        app.main_view.grid(column=1, row=0, sticky="news", padx=10, pady=10)

    def start_recognition(self) -> None:
        gesture_recognition.start()

    def register_gesture(self) -> None:
        if self.register_popup is None or not self.register_popup.winfo_exists():
            self.register_popup = self.RegisterGesturePopUp(master=self)
        else:
            self.register_popup.focus()

    class RegisterGesturePopUp(ctk.CTkToplevel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.title("Gesture Registeration Options")
            self.geometry("500x300")

            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
            self.rowconfigure(1, weight=0)

            self.confirmation_frame = ctk.CTkFrame(self, height=50)
            self.confirmation_frame.pack_propagate(False)

            self.confirm_button = ctk.CTkButton(
                self.confirmation_frame,
                text="Confirm",
                fg_color="#333333",
                hover_color="#373737",
                font=("Century", 16),
                command=self.confirm,
            )

            self.cancel_button = ctk.CTkButton(
                self.confirmation_frame,
                text="Cancel",
                fg_color="#333333",
                hover_color="#373737",
                font=("Century", 16),
                command=self.cancel,
            )

            self.options_frame = ctk.CTkFrame(self)
            self.options_frame.pack_propagate(False)

            self.gesture_name = ctk.CTkEntry(
                self.options_frame,
                width=180,
                placeholder_text="Gesture name...",
            )

            self.frames_label = ctk.CTkLabel(
                self.options_frame,
                text="Adjust the slider below to change the registeration frame length :-",
                font=("Century", 12),
            )

            self.frames_slider = ctk.CTkSlider(
                self.options_frame,
                from_=90,
                to=210,
                number_of_steps=4,
            )

            self.frames_slider_value = ctk.StringVar(value="150 Frames")
            self.frames_slider_value_label = ctk.CTkLabel(
                self.options_frame,
                textvariable=self.frames_slider_value,
                font=("Century", 12),
            )

            self.confirm_button.pack(padx=10, side="right")
            self.cancel_button.pack(padx=10, side="right")
            self.confirmation_frame.grid(
                column=0, row=1, sticky="wes", padx=10, pady=(0, 10)
            )

            self.gesture_name.pack(pady=20)

            self.frames_label.pack()
            self.frames_slider.pack(pady=(5, 5))
            self.frames_slider_value_label.pack()

            self.frames_slider.bind(
                "<B1-Motion>",
                lambda _: self.frames_slider_value.set(
                    f"{int(self.frames_slider.get())} Frames"
                ),
            )

            self.options_frame.grid(column=0, row=0, sticky="nsew", padx=10, pady=10)

        def confirm(self) -> None:
            name = self.gesture_name.get()
            frames = self.frames_slider.get()

            self.destroy()
            hand_landmarker.register_hand_gesture(
                name,
                frames=int(frames),
            )

            app.sidebar.reload_scrollbar()

        def cancel(self) -> None:
            self.destroy()


class MainView(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.pack_propagate(False)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        self.image = ctk.CTkLabel(
            self,
            text="",
            height=150,
            image=ctk.CTkImage(Image.open(r"images/thumbnail.png"), size=(840, 150)),
        )

        self.description = ctk.CTkLabel(
            self,
            text="",
            image=ctk.CTkImage(Image.open(r"images/description.png"), size=(840, 320)),
        )

        self.image.grid(row=0, sticky="nsew", padx=10, pady=(10, 5))
        self.description.grid(row=1, sticky="nsew", padx=10, pady=(5, 10))


class GestureOptions(ctk.CTkFrame):
    def __init__(self, master, gesture: GestureData, **kwargs):
        super().__init__(master, **kwargs)
        self.pack_propagate(False)
        self.gesture = gesture

        self.gesture_name = ctk.CTkLabel(
            self,
            text=f"{gesture['GestureName']}",
            font=("Century", 20),
        )

        self.graph_button = ctk.CTkButton(
            self,
            text="Open Graph",
            font=("Century", 18),
            bg_color="transparent",
            fg_color="#333333",
            hover_color="#373737",
            height=40,
            command=self.show_graph,
        )

        self.delete_button = ctk.CTkButton(
            self,
            text="Delete Gesture",
            font=("Century", 18),
            bg_color="transparent",
            fg_color="#333333",
            hover_color="#373737",
            height=40,
            command=self.delete,
        )

        self.rename_button = ctk.CTkButton(
            self,
            text="Rename Gesture",
            font=("Century", 18),
            bg_color="transparent",
            fg_color="#333333",
            hover_color="#373737",
            height=40,
            command=self.rename,
        )

        self.redo_button = ctk.CTkButton(
            self,
            text="Redo Gesture",
            font=("Century", 18),
            bg_color="transparent",
            fg_color="#333333",
            hover_color="#373737",
            height=40,
            command=self.redo,
        )

        self.gesture_name.pack(side="left", padx=(20, 10))
        self.graph_button.pack(side="right", padx=(10, 10))
        self.redo_button.pack(side="right", padx=(10, 10))
        self.rename_button.pack(side="right", padx=(10, 10))
        self.delete_button.pack(side="right", padx=(10, 10))

    def show_graph(self) -> None:
        self.gesture["Graph"].show()

    def delete(self) -> None:
        os.remove(f"landmarks/{self.gesture['GestureName']}.pkl")
        app.sidebar.reload_scrollbar()

        app.main_view = MainView(app)
        app.main_view.grid(column=1, row=0, sticky="news", padx=10, pady=10)

    def rename(self) -> None:
        input_dialog = ctk.CTkInputDialog(
            text="Please input a gesture name below...",
            title="Rename Gesture",
            button_fg_color="#333333",
            button_hover_color="#373737",
        )

        input_ = input_dialog.get_input()
        if input_ == None:
            return

        gesture_data: GestureData = GestureData(
            GestureName=input_,
            LandmarksData=self.gesture["LandmarksData"],
            Frames=self.gesture["Frames"],
            Images=self.gesture["Images"],
            Graph=self.gesture["Graph"],
        )

        os.remove(f"landmarks/{self.gesture['GestureName']}.pkl")
        with open(f"landmarks/{input_}.pkl", "wb") as file:
            pickle.dump(gesture_data, file)

        app.sidebar.reload_scrollbar()
        app.main_view = MainView(app)
        app.main_view.grid(column=1, row=0, sticky="news", padx=10, pady=10)

    def redo(self) -> None:
        hand_landmarker.register_hand_gesture(
            self.gesture["GestureName"], self.gesture["Frames"]
        )

        app.sidebar.reload_scrollbar()
        app.main_view = MainView(app)
        app.main_view.grid(column=1, row=0, sticky="news", padx=10, pady=10)


class FramesScrollbar(ctk.CTkScrollableFrame):
    def __init__(self, master, gesture: GestureData, **kwargs):
        super().__init__(
            master,
            label_text="Frames",
            label_font=("Century", 22),
            label_fg_color="#333333",
            orientation="horizontal",
            **kwargs,
        )

        self.pack_propagate(False)
        self.rowconfigure(0, weight=1)

        self.frames = [
            ctk.CTkLabel(
                self,
                text="",
                image=ctk.CTkImage(Image.fromarray(image), size=(452, 339)),
            )
            for image in gesture["Images"]
        ]

        [
            frame.grid(column=index, row=0, sticky="wes", padx=(0, 10), pady=5)
            for index, frame in enumerate(self.frames)
        ]


class GestureView(ctk.CTkFrame):
    def __init__(self, master, gesture: GestureData, *args, **kwargs):
        super().__init__(
            master, bg_color="transparent", fg_color="transparent", *args, **kwargs
        )
        self.pack_propagate(False)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        self.buttons = GestureOptions(self, gesture, height=60)
        self.buttons.grid(column=0, row=0, sticky="wes", padx=10, pady=(10, 5))

        self.frames = FramesScrollbar(self, gesture)
        self.frames.grid(column=0, row=1, sticky="news", padx=10, pady=(5, 10))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{1100}x{580}")
        self.title("Hand Gesture Recognition System")

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        self.sidebar = SideBar(self, width=200)
        self.sidebar.grid(column=0, row=0, sticky="nsw", pady=(0, 10))

        self.main_view = MainView(self)
        self.main_view.grid(column=1, row=0, sticky="news", padx=10, pady=10)

        self.bottombar = BottomBar(self, height=60)
        self.bottombar.grid(column=1, row=1, sticky="wes", padx=20)

        self.bottom_left_frame = ctk.CTkFrame(self)
        self.bottom_left_frame.pack_propagate(False)

        self.scaling_label = ctk.CTkLabel(
            self.bottom_left_frame, text="UI Scaling:", anchor="w"
        )

        self.scaling_label.pack()
        self.scaling_optionemenu = ctk.CTkOptionMenu(
            self.bottom_left_frame,
            values=["100%", "110%", "120%", "130%", "140%"],
            command=self.change_scaling_event,
            fg_color="#333333",
            button_color="#333333",
            button_hover_color="#373737",
        )

        self.scaling_optionemenu.pack(pady=(0, 5))
        self.bottom_left_frame.grid(row=1, column=0, sticky="news")

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)


app = App()
app.mainloop()
