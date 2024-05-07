import tensorflow as tf
from PredictionBuffer import PredictionBuffer

class CNNModel:
    def __init__(self, modelPath: str):
        self.model = tf.keras.models.load_model(modelPath)
        self.classes = []
        self.prediction_buffer = PredictionBuffer()

    def predict(self, image):
        prediction = self.model.predict(image)
        self.prediction_buffer.add_accuracy(prediction.max())
        return prediction

    def get_average_accuracy(self):
        return self.prediction_buffer.get_average_accuracy()

    def get_model(self):
        return self.model

    def restart_buffer(self):
        self.prediction_buffer.restart_buffer()

    def set_classes(self, classes: list):
        self.classes = classes

    def get_classes(self):
        return self.classes