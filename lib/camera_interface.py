import os
import threading
import cv2
import numpy as np
import time
from datetime import datetime

from lib.servo_interface import SerialInterface
from lib.cnn_classifier import MobileNetClassifier


class MainInterface:
    def __init__(
            self,
            data_dir='./data/src/',
            camera_index=1,
            marked_classes=['a'],
            model_path='./models/mobilenet_best.keras'
        ):
        self.running = True
        self._data_dir = data_dir
        self._camera_index = camera_index
        self._capture = cv2.VideoCapture(self._camera_index)
        self.ret = False
        self.text = "Classifying..."
        self._gui_thread = threading.Thread(target=self._main_loop)
        self._gui_thread.start()

        self.servo = SerialInterface()
        self.angle_deg = 45
        
        self.classifier = MobileNetClassifier()
        self.classifier.load_model(model_path)
        self._classifier_min_frequency_hz = 1
        self.marked_classes = marked_classes
        self._classifier_thread = threading.Thread(target=self._continuous_classification)
        self._classifier_thread.start()

    def _main_loop(self):
        while self.running:
            self.ret, self.frame = self._capture.read()
            if not self.ret:
                self.frame = np.zeros(shape=(200, 200, 3), dtype=np.uint8)
            cv2.putText(
                self.frame,
                self.text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow(f'Camera {self._camera_index}', self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self._save_picture(self.frame)
            elif key == ord('q'):
                self._terminate()

    def _continuous_classification(self):
        while self.running:
            if self.ret:
                prediction = self.classifier.predict(self.frame)
                if prediction == 'a':
                    self.text = "Garbage..."
                elif prediction == 'b':
                    self.text = "Money!"
                else:
                    self.text = "Who knows what is that..."
                if prediction in self.marked_classes:
                    self._swing()
            time.sleep(1 / self._classifier_min_frequency_hz) 

    def _swing(self):
        self.angle_deg = 45 if self.angle_deg == 135 else 135
        self.servo.run_sequence([self.angle_deg])

    def _save_picture(self, frame):
        if frame is not None:
            filename = os.path.join(
                self._data_dir,
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(filename, frame)

    def _terminate(self):
        self.running = False
        self._capture.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self._terminate()
