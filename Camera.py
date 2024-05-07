import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import threading

from CNNModel import CNNModel
from Keyboard import Keyboard

class Camera:
    def __init__(self):
        self.camera = cv.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.predicted_class = None
        self.last_predicted_class = None
        self.prob = 0.0
        self.keyboard = None
        self.last_key_press_time = 0
        self.key_pressed_recently = False
        self.key_press_timer = None
        self.key_press_lock = threading.Lock()

    def _predict_thread(self, image, hand_landmarks, image_shape, model):
        image_to_the_model = self._capture_hand_image(image, hand_landmarks, image_shape)
        if image_to_the_model is not None:
            self._predict_image(image_to_the_model, model)

        time.sleep(0.1)

    def _capture_hand_image(self, image, hand_landmarks, original_image_shape):
        if hand_landmarks is not None and len(hand_landmarks) == 1:
            x_min, y_min, x_max, y_max = self._get_bounding_box(hand_landmarks[0], original_image_shape)
            cropped_image = image[y_min:y_max, x_min:x_max]
            # Resize the single-hand image to a fixed size
            resized_image = cv.resize(cropped_image, (224, 224))
            return resized_image
        elif hand_landmarks is not None and len(hand_landmarks) == 2:
            right_x_min, right_y_min, right_x_max, right_y_max = self._get_bounding_box(hand_landmarks[1],
                                                                                  original_image_shape)
            right_hand_cropped_image = image[right_y_min:right_y_max, right_x_min:right_x_max]

            # Calculate maximum height among the two hands
            max_height = right_hand_cropped_image.shape[0]

            # Resize right hand image to match maximum height
            right_hand_resized = cv.resize(right_hand_cropped_image, (
            int(right_hand_cropped_image.shape[1] * max_height / right_hand_cropped_image.shape[0]), max_height))

            # Resize the combined image to a fixed size
            resized_image = cv.resize(right_hand_resized, (224, 224))
            return resized_image
        elif hand_landmarks is None:
            print("No hand landmarks detected")
            return None

    def _predict_image(self, image, model):
        # image = cv.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        if self.last_predicted_class != model.get_classes()[np.argmax(prediction)]:
            self.predicted_class = model.get_classes()[np.argmax(prediction)]
            self.last_predicted_class = self.predicted_class
        self.prob = float(prediction.max())
        if model.get_average_accuracy() > np.float64(0.90):
            if self.keyboard is not None:
                if len(self.predicted_class) == 1 and self.predicted_class.isalpha():
                    if self.predicted_class.isupper():
                        current_time = time.time()
                        with self.key_press_lock:
                            if not self.key_pressed_recently and current_time - self.last_key_press_time >= 2:
                                self.keyboard.pressKey(str(self.predicted_class))
                                self.last_key_press_time = current_time
                                self.key_pressed_recently = True
                                self._start_timer()
                                model.restart_buffer()

    def _start_timer(self):
        """
        Start a timer to reset the key pressed flag after 2 seconds.
        """
        if self.key_press_timer and self.key_press_timer.is_alive():
            self.key_press_timer.cancel()
        self.key_press_timer = threading.Timer(2, self._reset_key_pressed_recently)
        self.key_press_timer.start()

    def _reset_key_pressed_recently(self):
        """
        Reset the key pressed flag.
        """
        self.key_pressed_recently = False

    def _get_bounding_box(self, hand_landmarks, image_shape):
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        x_min = int(min(x_coords) * image_shape[1])
        x_max = int(max(x_coords) * image_shape[1])
        y_min = int(min(y_coords) * image_shape[0])
        y_max = int(max(y_coords) * image_shape[0])
        return x_min, y_min, x_max, y_max

    def start_camera(self, model: CNNModel):
        with self.mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.camera.isOpened():
                success, image = self.camera.read()
                original_image_shape = image.shape
                start = time.time()

                image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get bounding box coordinates around the hand landmarks
                        x_min, y_min, x_max, y_max = self._get_bounding_box(hand_landmarks, original_image_shape)

                        # Draw green rectangle around the bounding box
                        cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv.putText(image, f"{self.predicted_class} {self.prob * 100:.2f}%", (150, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Start a new thread for making predictions
                prediction_thread = threading.Thread(target=self._predict_thread, args=(
                image, results.multi_hand_landmarks, original_image_shape, model))
                prediction_thread.start()

                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime
                cv.putText(image, f"FPS: {fps:.3f}", (30, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv.LINE_AA)
                cv.imshow("Image", image)

                key = cv.waitKey(5)
                if key == 27:
                    break

        cv.destroyAllWindows()
        self.camera.release()

    def set_keyboard(self, keyboard: Keyboard):
        self.keyboard = keyboard